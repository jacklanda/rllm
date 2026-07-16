"""Project-owned workflow entrypoint for eval rollouts."""

from __future__ import annotations

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.types import Episode
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason


class EvalsWorkflow(MultiTurnWorkflow):
    """Multi-step workflow that preserves raw model outputs on each step."""

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0, False, info)

        for _ in range(1, self.max_steps + 1):
            
            #llm_kwargs = dict(kwargs)
            #if "tools" not in llm_kwargs:
            #    multi_tool = getattr(self.env, "multi_tool", None)
            #    if multi_tool is not None:
            #        tool_schemas = multi_tool.json
            #        if tool_schemas:
            #            llm_kwargs["tools"] = tool_schemas

            output: ModelOutput = await self.timed_llm_call(
                self.agent.chat_completions,
                application_id=uid,
                **kwargs,
                #**llm_kwargs,
            )
            response = output.text or ""
            action = self.agent.update_from_model(response, model_output=output)

            next_obs, reward, done, info = await self.timed_env_call(
                self.env.step, action
            )
            self.agent.update_from_env(next_obs, reward, done, info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)


__all__ = ["EvalsWorkflow"]
