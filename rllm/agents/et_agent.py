"""Endless Terminals agent.

Reuses ``SWEAgent``'s function-call protocol (XML ``<function=...>`` parsing,
trajectory bookkeeping, message construction) and only swaps the system /
user prompt templates so the agent is framed as a CLI agent operating in
``/home/user`` rather than as a SWE-Bench agent operating on ``/testbed``.
"""

from rllm.agents.swe_agent import SWEAgent, parse_oai_response
from rllm.agents.system_prompts import ET_AGENT_SYSTEM_PROMPT, ET_AGENT_USER_PROMPT
from rllm.agents._dual_parser import parse_dual_response


class ETAgent(SWEAgent):
    """Agent for Endless Terminals tasks.

    Identical chat surface to SWEAgent (so it remains compatible with the
    existing rollout/workflow plumbing) but with ET-specific prompts.

    Overrides ``_parse_response`` to accept both SWE-XML
    ``<function=...>`` calls and Qwen-native ``<tool_call>`` JSON, since
    thinking-base models often emit the latter regardless of system-prompt
    instructions. SWE behaviour is unaffected (this only kicks in for
    ETAgent instances).
    """

    def __init__(self, use_fn_calling: bool = False, format_model_response: bool = False, **kwargs):
        # SWEAgent's __init__ accepts ``scaffold`` for r2egym vs sweagent tools;
        # ET only ships the sweagent tool variants (str_replace_editor +
        # execute_bash + submit), so force scaffold="sweagent". Drop any
        # caller-provided scaffold to keep things simple.
        kwargs.pop("scaffold", None)
        super().__init__(
            use_fn_calling=use_fn_calling,
            format_model_response=format_model_response,
            scaffold="sweagent",
        )
        # SWEAgent picks SWEAGENT_* by virtue of scaffold="sweagent"; replace
        # with the ET equivalents AFTER super().__init__ has reset the
        # trajectory and seeded self.messages with the system prompt.
        self.system_prompt = ET_AGENT_SYSTEM_PROMPT
        self.user_prompt_template = ET_AGENT_USER_PROMPT
        self.reset()

    def _parse_response(self, response: str):
        if self.use_fn_calling:
            return parse_oai_response(response)
        return parse_dual_response(response)
