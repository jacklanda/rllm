"""Compatibility agent/environment for the current rLLM workflow API."""

from __future__ import annotations

from typing import Any
import json
from evals.tokenizer_compat import get_tool_parser_compat
from rllm.agents.agent import BaseAgent
from rllm.environments.base.base_env import BaseEnv
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import ToolCall, ToolOutput
from rllm.types import Action, Step, Trajectory
import re


_BOXED_BACKSLASH_RE = re.compile(r"(?<!\\)((?:\\\\)*\\)boxed")


def _escape_unescaped_boxed_for_tool_parse(text: str) -> str:
    """Make LaTeX boxed markers JSON-safe before tool-call parsing.

    Models often emit tool-call JSON containing ``\boxed{...}``. A single
    backslash is not a valid JSON escape, so parsers can reject an otherwise
    usable finish call. This only touches odd-backslash ``boxed`` markers and
    leaves already escaped ``\\boxed`` markers unchanged.
    """
    if "\\boxed" not in (text or ""):
        return text
    return _BOXED_BACKSLASH_RE.sub(lambda m: f"{m.group(1)}\\boxed", text)


class EvalsAgent(BaseAgent):
    """Small chat agent compatible with rLLM's legacy MultiTurnWorkflow."""

    def __init__(
        self,
        system_prompt: str = "",
        user_prompt_template: str = "{problem_statement}",
        parser_name: str = "qwen",
        model: str | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, Any] | None = None,
        is_no_think_prompt: bool = False,
        **_: Any,
    ) -> None:
        self.base_system_prompt = system_prompt or ""
        self.system_prompt = self.base_system_prompt
        self.user_prompt_template = user_prompt_template
        self.parser_name = parser_name or "qwen"
        self.is_no_think_prompt = is_no_think_prompt
        self.tools = list(tools or [])
        self.tool_map = dict(tool_map or {})
        self._parser = get_tool_parser_compat(self.parser_name, model=model)
        self.reset()

    def reset(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._trajectory = Trajectory()
        self._pending_model_response = ""
        self._pending_action: Any = None
        self._pending_model_output: Any = None

    @property
    def chat_completions(self) -> list[dict[str, Any]]:
        return list(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(
        self, observation: Any, reward: float, done: bool, info: dict, **_: Any
    ) -> None:
        if not self._messages:
            system_prompt = self.base_system_prompt

            tools_json = (info or {}).get("tools_json") or []
            if tools_json and hasattr(self._parser, "get_tool_prompt"):
                schemas_str = "\n".join(
                    json.dumps(schema, indent=0, ensure_ascii=False)
                    for schema in tools_json
                )
                system_prompt = (
                    system_prompt.rstrip()
                    + "\n"
                    + self._parser.get_tool_prompt(schemas_str)
                )
            
            if system_prompt:
                self._messages.append({"role": "system", "content": system_prompt})
            question = ""
            if isinstance(observation, dict):
                question = str(observation.get("question") or observation.get("prompt") or "")
            else:
                question = str(observation or "")
            user_prompt = self.user_prompt_template.format(
                problem_statement=question,
            )

            self._messages.append({"role": "user", "content": user_prompt})
            return

        if observation not in (None, "", {}, []):
            self._messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{observation}\n</tool_response>",
                }
            )

        self._trajectory.steps.append(
            Step(
                chat_completions=list(self._messages),
                observation=observation,
                action=self._pending_action,
                model_response=self._pending_model_response,
                model_output=self._pending_model_output,
                thought=getattr(self._pending_model_output, "reasoning", "") or "",
                reward=float(reward),
                done=bool(done),
                metadata=dict(info or {}),
            )
        )


    def _strip_think(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


    def update_from_model(
        self, response: str, model_output: Any = None, **_: Any
    ) -> Action:
        response = response or ""
        self._pending_model_response = response
        self._pending_model_output = model_output

        content = getattr(model_output, "content", None) if model_output else None
        reasoning = getattr(model_output, "reasoning", None) if model_output else None
        raw_text = getattr(model_output, "text", None) if model_output else response
        history_content = raw_text if raw_text is not None else response
        #print('-' * 30)
        #print(history_content)
        #print('-' * 30)
        if self.is_no_think_prompt:
            history_content = self._strip_think(history_content)
            
        #print('-' * 30)
        #print(history_content)
        #print('-' * 30)
            

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": history_content,
        }
        #print(assistant_message)
        #if reasoning:
        #    assistant_message["reasoning_content"] = reasoning
        #raw_text = getattr(model_output, "text", None) if model_output else response
        #if raw_text and raw_text != assistant_message["content"]:
        #    assistant_message["raw_content"] = raw_text

        #tool_calls = getattr(model_output, "tool_calls", None) if model_output else None
        #if tool_calls:
        #    assistant_message["tool_calls"] = [
        #        {
        #            "function": {
        #                "name": getattr(call, "name", ""),
        #                "arguments": getattr(call, "arguments", {}) or {},
        #            }
        #        }
        #        for call in tool_calls
        #    ]

        self._messages.append(assistant_message)

        parse_response = _escape_unescaped_boxed_for_tool_parse(response)
        tool_calls = self._parser.parse(parse_response)
        raw_action: Any = tool_calls if tool_calls else response
        self._pending_action = Action(action=raw_action)
        return self._pending_action


class EvalsEnvironment(BaseEnv):
    """QA environment with finish-tool scoring and optional rLLM tools."""

    def __init__(
        self,
        reward_fn,
        max_steps: int = 10,
        tools: list[str] | None = None,
        tool_map: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.multi_tool = MultiTool(tools=tools, tool_map=tool_map) if (tools or tool_map) else None
        self.task: dict[str, Any] = {}
        self.num_steps = 0

    def reset(self, task: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.task = dict(task or {})
        self.num_steps = 0
        info = {}
        if self.multi_tool is not None:
            info["tools_json"] = self.multi_tool.json

        return {"question": self.task.get("question", "")}, info

        #return {"question": self.task.get("question", "")}, {}

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.num_steps += 1
        raw_action = action.action if isinstance(action, Action) else action

        if isinstance(raw_action, list):
            observations: list[str] = []
            final_answer = None
            for call in raw_action:
                if isinstance(call, ToolCall) and call.name == "finish":
                    final_answer = _extract_finish_answer(call.arguments)
                    break
                observations.append(str(self._run_tool(call)))
            if final_answer is not None:
                return self._score(final_answer)
            done = self.num_steps >= self.max_steps
            return "\n".join(observations), 0.0, done, {}

        return self._score(str(raw_action or ""))

    def _run_tool(self, call: Any) -> ToolOutput:
        if not isinstance(call, ToolCall):
            return ToolOutput(name="unknown", error=f"Unsupported action: {call!r}")
        if self.multi_tool is None:
            return ToolOutput(name=call.name, error="No tools configured")
        return self.multi_tool.forward(tool_name=call.name, **(call.arguments or {}))

    def _score(self, answer: str) -> tuple[str, float, bool, dict]:
        result = self.reward_fn(self.task, answer)
        info = {
            "is_correct": bool(getattr(result, "is_correct", False)),
            "metadata": dict(getattr(result, "metadata", {}) or {}),
        }
        return "", float(getattr(result, "reward", 0.0)), True, info

    @staticmethod
    def from_dict(info: dict) -> "EvalsEnvironment":
        return EvalsEnvironment(**info)


def _extract_finish_answer(arguments: dict[str, Any] | None) -> str:
    args = arguments or {}
    for key in ("answer", "final_answer", "response", "result"):
        if key in args:
            return str(args[key])
    return str(args)


__all__ = ["EvalsAgent", "EvalsEnvironment"]
