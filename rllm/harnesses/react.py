"""ReActHarness: LLM calls with optional Qwen-style tool use.

Default harness for catalog datasets (gsm8k, MATH, MMLU, etc.) where the
agent can usually answer with a single chat completion. When a task provides
tool schemas and callables in metadata, this harness runs a standard ReAct
loop: reason, call tools with ``<tool_call>`` blocks, observe tool results,
and continue until the model stops calling tools.

For sandbox tasks (Harbor, SWE-bench), use :class:`rllm.harnesses.bash.BashHarness`
instead. This harness has no sandbox dependency.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from rllm.parser.tool_parser import QwenToolParser
from rllm.tools.tool_base import Tool, ToolCall
from rllm.types import Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions by reasoning and searching for relevant information.

Follow the ReAct format strictly:
- Thought: reason about what you know and what you need to find next
- Action: call a tool with appropriate arguments
- Observation: the tool result (provided by the environment)

Repeat Thought/Action/Observation until you have enough information to answer. Then call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""


class ReActHarness:
    """LLM harness for data tasks with optional tool use.

    If no executable tools are supplied in task metadata, this behaves like
    the old one-shot harness. If tools are supplied, it runs a bounded ReAct
    loop and appends tool outputs as ``<tool_response>`` user messages.
    """

    name = "react"
    max_concurrent = 64

    def __init__(self, system_prompt: str | None = None, max_turns: int = 8):
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.max_turns = max_turns
        self.tool_parser = QwenToolParser()

    def run(self, task: Task, config) -> Episode:
        from openai import OpenAI

        try:
            from rllm.eval.reward_fns._resolver import get_verifier_system_prompt
        except ModuleNotFoundError:
            get_verifier_system_prompt = lambda _task: None

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        tools_json, tool_map = _resolve_task_tools(task)
        if tool_map:
            self.tool_parser.valid_tools = set(tool_map) | {"finish", "submit"}

        system_msg = self.system_prompt
        if tools_json:
            tools_schema = "\n".join(json.dumps(t, indent=0, ensure_ascii=False) for t in tools_json)
            system_msg = f"{system_msg}\n\n{self.tool_parser.get_tool_prompt(tools_schema)}"

        verifier_hint = get_verifier_system_prompt(task)
        if verifier_hint:
            system_msg = f"{system_msg}\n\n{verifier_hint}"

        instruction = task.instruction
        user_content = instruction if isinstance(instruction, list) else str(instruction)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        steps: list[Step] = []
        all_tool_calls: list[dict[str, Any]] = []
        answer = ""
        max_turns = int(task.metadata.get("rllm", {}).get("max_turns") or self.max_turns)

        for turn in range(max_turns):
            try:
                response = client.chat.completions.create(model=config.model, messages=messages)
                assistant_msg = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("ReActHarness LLM call failed for task %s: %s", task.id, e)
                assistant_msg = ""

            messages.append({"role": "assistant", "content": assistant_msg})
            answer = assistant_msg
            tool_calls = self.tool_parser.parse(assistant_msg)
            all_tool_calls.extend(_tool_call_to_dict(tc) for tc in tool_calls)

            step = Step(
                id=f"step-{turn}",
                input=str(instruction) if turn == 0 and not isinstance(instruction, list) else ("<multimodal>" if turn == 0 else ""),
                output=assistant_msg,
                action=[_tool_call_to_dict(tc) for tc in tool_calls],
                done=not tool_calls,
            )
            steps.append(step)

            if not tool_calls:
                break

            # BFCL-style tasks may only require emitting function calls; if no
            # executable callables were supplied, stop after recording them.
            if not tool_map:
                break

            observations = []
            for tool_call in tool_calls:
                output = _execute_tool_call(tool_call, tool_map)
                observations.append(f"{tool_call.name}: {output}")
            messages.append(
                {
                    "role": "user",
                    "content": f"{self.tool_parser.tool_output_begin}\n" + "\n\n".join(observations) + f"\n{self.tool_parser.tool_output_end}",
                }
            )

        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=steps,
            output=answer,
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
            artifacts={
                "answer": answer,
                "tool_calls": all_tool_calls,
            },
        )


def _resolve_task_tools(task: Task) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = task.metadata or {}
    tools_json = list(metadata.get("tools_json") or metadata.get("tools") or [])
    tool_map = dict(metadata.get("tool_map") or metadata.get("callables") or {})

    for name, tool in list(tool_map.items()):
        if isinstance(tool, Tool):
            tools_json.append(tool.json)
        elif callable(tool):
            tools_json.append(Tool(function=tool).json)
        else:
            logger.warning("Ignoring non-callable tool %s on task %s", name, task.id)
            tool_map.pop(name, None)

    return tools_json, tool_map


def _execute_tool_call(tool_call: ToolCall, tool_map: dict[str, Any]) -> str:
    tool = tool_map.get(tool_call.name)
    if tool is None:
        return f"Error: unknown tool {tool_call.name}"

    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    try:
        if isinstance(tool, Tool):
            return tool.forward(**args).to_string()
        if callable(tool):
            result = tool(**args)
            return json.dumps(result, ensure_ascii=False) if isinstance(result, list | dict) else str(result)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return f"Error: unsupported tool {tool_call.name}"


def _tool_call_to_dict(tool_call: ToolCall) -> dict[str, Any]:
    return {"name": tool_call.name, "arguments": tool_call.arguments}
