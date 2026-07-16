"""Harness registry for agentic evaluation.

Each harness bundles a system prompt, tool-call parser, and tool list into a
named configuration. Select one via ``--harness <name>`` on the CLI; explicit
``--prompt``, ``--parser-name``, or ``--tools`` flags override the harness
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration bundle for an agent harness."""

    name: str
    system_prompt: str
    user_prompt_template: str = "{problem_statement}"
    parser_name: str = "qwen"
    tools: tuple[str, ...] = ("web_search",)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_FUSED_SEARCH_SYSTEM_PROMPT = """\
You are a research assistant that answers questions by searching for relevant information. You have access to a web_search tool for looking up facts, and a finish tool to submit your final answer.

RULES:
1. Call web_search as many times as needed — keep searching until you have concrete evidence (named entities, dates, numbers) for every part of the question. Do NOT stop after a fixed number of searches.
2. If a search result is short, vague, or only echoes the query, issue a new query with different keywords — never submit based on low-content results.
3. For multi-hop questions, decompose into sub-questions and search each sub-question separately.
4. Use the same language as the question when you write queries (e.g., Chinese question -> Chinese query).
5. Synthesize the search results to form an accurate, concise answer. Only submit once you can ground each claim in retrieved text.
6. Your final answer should be clearly stated in \\boxed{} format inside the finish tool's ``result``.
"""

_REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by interleaving reasoning and tool use in a loop.

At each step:
1. Think: Briefly analyze what you know so far and decide what information you still need.
2. Act: Call exactly one tool to retrieve the needed information.
3. Observe: Read the tool result, then return to step 1.

Guidelines:
- Each thought should be concise (2-4 sentences). State what you learned from the last observation and what gap remains.
- Choose search queries that are specific and likely to return relevant results. If a query returns nothing useful, reformulate at a different level of specificity rather than repeating.
- Do not call the same tool with the same arguments twice.
- Stop searching when you have sufficient evidence. Do not over-search.
- When you have enough information to answer confidently, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""

_GEM_SYSTEM_PROMPT = """\
You are a helpful AI assistant that answers questions by progressively searching for relevant information.

When answering the question:
1. Use the search tool to find relevant information and synthesize them from multiple sources when needed.
2. You are asked to perform search only once per turn. Each time you search, think about what you need to find next turn based on what you have already found.
3. Use the search tool instead of relying on your own knowledge. You must perform at least 3 turns of search tool calls before concluding your answer.
4. Your middle turns must contain a valid tool call. Your final turn must contain the final answer.

When you have searched thoroughly and have sufficient evidence, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter.

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}"""

_AWM_SYSTEM_PROMPT = """\
You are at a tool-use environment. You need to call tools to assist with the user query. At each step, you can only call one function.

Important rules:
- Each tool call must be purposeful. Track what you have learned and what remains unknown.
- Do not repeat queries that have already been answered.
- You should always directly output the answer or summary at the final step instead of calling any function.
- When you have gathered sufficient evidence, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""

_SIMIA_SYSTEM_PROMPT = """\
You are an AI assistant that completes user tasks through multi-turn tool-use interactions.

You must reason explicitly before each tool call. For each function call, first provide brief reasoning (1-3 sentences) inside <think> </think> tags explaining why this function call is needed and what you expect to learn. End your reasoning with: I will call the function [function_name]. Then make the function call.

Requirements:
- Always reason before acting in <think> tags. Do not call tools without reasoning first.
- Reasoning should ONLY appear immediately before a function call, not in other parts of your response.
- Use the minimum necessary tool calls to complete the task.
- Do not repeat failed queries without adjusting your approach.
- You should always directly output the answer at the final step instead of calling any function. When you have sufficient information, call the finish tool with your final answer in \\boxed{} format inside the result parameter."""

_RLVE_SYSTEM_PROMPT = """\
You are a helpful assistant that solves problems through step-by-step reasoning.

Show your reasoning process in <think> </think> tags. Break the problem into sub-problems of manageable difficulty. Start with what you can determine, then build toward harder sub-problems. Verify each step before proceeding to the next.

You have access to a search tool. Use it when you need external information, but rely on your own reasoning when the problem can be solved through logical deduction. If a search returns insufficient results, reformulate your query at a different level of specificity.

When you have verified sufficient evidence to answer confidently, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""

_ENVSCALER_SYSTEM_PROMPT = """\
You are at a tool-use environment. You need to call tools to assist with the user query. At each step, you can only call one function.

Important rules:
- Each tool call must be purposeful. Track what you have learned and what remains unknown.
- Do not repeat queries that have already been answered.
- You should always directly output the answer or summary at the final step instead of calling any function.
- When you have gathered sufficient evidence, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""

_TOUCAN_SYSTEM_PROMPT = """\
You are an AI assistant that completes user tasks by orchestrating tool calls. You have access to tools for information retrieval.

You may need to call multiple tools in sequence to complete a task. Plan your tool calls to cover the information needed, adapt based on results, and synthesize findings from multiple calls into a complete answer.

Requirements:
- Each tool call should target specific information needed to answer the question.
- If a tool call returns an error or insufficient results, adapt your approach rather than repeating the same query.
- If the available tools cannot solve the user's question, say so directly instead of hallucinating an answer.
- Synthesize information from multiple tool calls to form a complete answer.
- Be concise and efficient — use the minimum number of tool calls necessary.
- When you have gathered sufficient information, call the finish tool with your final answer clearly stated in \\boxed{} format inside the result parameter."""

_COT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\\\boxed{}."
)
#----------------------------------------------------------------------------
# User prompt
#----------------------------------------------------------------------------
_FUSED_SEARCH_USER_PROMPT = """Answer the following question by searching for relevant information.

<question>
{problem_statement}
</question>

Instructions:
1. Use the web_search tool to find relevant information. Search as many times as needed and do not stop after a fixed number of searches — keep querying until you have concrete supporting evidence.
2. If a search result is short, vague, or just echoes the question, issue a new query with different keywords or add named entities, dates, or numbers.
3. For multi-hop questions, decompose into sub-questions and search each separately.
4. Write queries in the same language as the question (e.g., Chinese question -> Chinese query).
5. Synthesize the search results to form an accurate answer grounded in retrieved text.
6. When you have found the answer, use the finish tool to submit your response with your answer in the result parameter.
7. Your final answer should also be clearly stated in \\boxed{{}} format.

IMPORTANT: Do NOT use file editing tools (file_editor, execute_bash, search) for this task — only use web_search and finish.
"""
# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HARNESS_REGISTRY: dict[str, HarnessConfig] = {
    "react": HarnessConfig(
        name="react",
        system_prompt=_REACT_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    #"gem": HarnessConfig(
    #    name="gem",
    #    system_prompt=_GEM_SYSTEM_PROMPT,
    #    parser_name="qwen",
    #    tools=("web_search",),
    #),
    "gem_new": HarnessConfig(
        name="gem_new",
        system_prompt=_FUSED_SEARCH_SYSTEM_PROMPT,
        user_prompt_template=_FUSED_SEARCH_USER_PROMPT,
        parser_name="qwen",
        tools=("web_search", "finish"),
    ),
    "awm": HarnessConfig(
        name="awm",
        system_prompt=_AWM_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    "simia": HarnessConfig(
        name="simia",
        system_prompt=_SIMIA_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    "rlve": HarnessConfig(
        name="rlve",
        system_prompt=_RLVE_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    "envscaler": HarnessConfig(
        name="envscaler",
        system_prompt=_ENVSCALER_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    "toucan": HarnessConfig(
        name="toucan",
        system_prompt=_TOUCAN_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=("web_search",),
    ),
    "cot": HarnessConfig(
        name="cot",
        system_prompt=_COT_SYSTEM_PROMPT,
        parser_name="qwen",
        tools=(),
    ),
}


def get_harness(name: str) -> HarnessConfig:
    """Look up a harness by name. Raises KeyError if not found."""
    if name not in HARNESS_REGISTRY:
        raise KeyError(
            f"Unknown harness {name!r}. Available: {sorted(HARNESS_REGISTRY)}"
        )
    return HARNESS_REGISTRY[name]
