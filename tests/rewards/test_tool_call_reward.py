#!/usr/bin/env python3
"""
Test script to verify the tool call bonus reward implementation.
"""

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


def test_tool_call_parsing():
    """Test the parse_tool_calls method with various inputs."""
    config = RewardConfig(toolcall_bonus=0.5)
    reward_fn = RewardSearchFn(config)

    # Test case 1: Single valid tool call
    response1 = """
    Let me search for information.
    <tool_call>
    {"name": "local_search", "arguments": {"query": "test query"}}
    </tool_call>
    The result shows that the answer is 42.
    """
    count1, matches1, valid1 = reward_fn.parse_tool_calls(response1)
    print(f"Test 1 - Single tool call:")
    print(f"  Count: {count1}, Expected: 1")
    print(f"  Matches: {len(matches1)}")
    print(f"  Valid: {valid1}, Expected: True")
    print()

    # Test case 2: Multiple tool calls (should be penalized)
    response2 = """
    <tool_call>
    {"name": "search", "arguments": {"query": "first"}}
    </tool_call>
    <tool_call>
    {"name": "search", "arguments": {"query": "second"}}
    </tool_call>
    """
    count2, matches2, valid2 = reward_fn.parse_tool_calls(response2)
    print(f"Test 2 - Multiple tool calls:")
    print(f"  Count: {count2}, Expected: 2")
    print(f"  Matches: {len(matches2)}")
    print(f"  Valid: {valid2}, Expected: True")
    print()

    # Test case 3: No tool call
    response3 = "Just a plain answer without any tool call."
    count3, matches3, valid3 = reward_fn.parse_tool_calls(response3)
    print(f"Test 3 - No tool call:")
    print(f"  Count: {count3}, Expected: 0")
    print(f"  Matches: {len(matches3)}")
    print(f"  Valid: {valid3}, Expected: True")
    print()

    # Test case 4: Mismatched tags (unclosed)
    response4 = """
    <tool_call>
    {"name": "search", "arguments": {"query": "test"}}
    No closing tag here!
    """
    count4, matches4, valid4 = reward_fn.parse_tool_calls(response4)
    print(f"Test 4 - Unclosed tool call:")
    print(f"  Count: {count4}, Expected: 0 (invalid)")
    print(f"  Matches: {len(matches4)}")
    print(f"  Valid: {valid4}, Expected: False")
    print()

    # Test case 5: Mismatched tags (extra closing)
    response5 = """
    <tool_call>
    {"name": "search"}
    </tool_call>
    </tool_call>
    """
    count5, matches5, valid5 = reward_fn.parse_tool_calls(response5)
    print(f"Test 5 - Extra closing tag:")
    print(f"  Count: {count5}, Expected: 0 (invalid)")
    print(f"  Matches: {len(matches5)}")
    print(f"  Valid: {valid5}, Expected: False")
    print()


def test_reward_calculation():
    """Test the full reward calculation with tool call bonus (NEW STRATEGY)."""
    config = RewardConfig(
        correct_reward=1.0,
        incorrect_reward=0.0,
        toolcall_bonus=0.5
    )
    reward_fn = RewardSearchFn(config)

    # Test case 1: Correct answer with single tool call (should get bonus)
    task_info1 = {"ground_truth": "42"}
    action1 = """
    <tool_call>
    {"name": "search", "arguments": {"query": "answer"}}
    </tool_call>
    Based on the search, the answer is **42**.
    """
    reward_input1 = RewardInput(task_info=task_info1, action=action1)
    result1 = reward_fn(reward_input1)
    print(f"Test 1 - Correct answer with single tool call (NEW: bonus only if reward > 0):")
    print(f"  Reward: {result1.reward} (Expected: 1.5, 1.0 + 0.5 bonus)")
    print(f"  Tool call count: {result1.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result1.metadata.get('tool_call_status')}")
    print(f"  Tool call adjustment: {result1.metadata.get('tool_call_adjustment')}")
    print()

    # Test case 2: Correct answer without tool call (no bonus)
    action2 = "The answer is **42**."
    reward_input2 = RewardInput(task_info=task_info1, action=action2)
    result2 = reward_fn(reward_input2)
    print(f"Test 2 - Correct answer without tool call:")
    print(f"  Reward: {result2.reward} (Expected: 1.0)")
    print(f"  Tool call count: {result2.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result2.metadata.get('tool_call_status')}")
    print()

    # Test case 3: Correct answer with multiple tool calls (NEW: -0.5 penalty always)
    action3 = """
    <tool_call>{"name": "search", "arguments": {"query": "first"}}</tool_call>
    <tool_call>{"name": "search", "arguments": {"query": "second"}}</tool_call>
    The answer is **42**.
    """
    reward_input3 = RewardInput(task_info=task_info1, action=action3)
    result3 = reward_fn(reward_input3)
    print(f"Test 3 - Correct answer with multiple tool calls (NEW: -0.5 penalty):")
    print(f"  Reward: {result3.reward} (Expected: 0.5, 1.0 - 0.5 penalty)")
    print(f"  Tool call count: {result3.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result3.metadata.get('tool_call_status')}")
    print(f"  Tool call adjustment: {result3.metadata.get('tool_call_adjustment')}")
    print()

    # Test case 4: Incorrect answer with tool call (NEW: NO bonus, avoid reward hacking)
    action4 = """
    <tool_call>{"name": "search"}</tool_call>
    The answer is wrong.
    """
    reward_input4 = RewardInput(task_info=task_info1, action=action4)
    result4 = reward_fn(reward_input4)
    print(f"Test 4 - Incorrect answer with tool call (NEW: NO bonus to avoid reward hacking):")
    print(f"  Reward: {result4.reward} (Expected: 0.0, no bonus for incorrect answer)")
    print(f"  Is correct: {result4.is_correct}")
    print(f"  Tool call count: {result4.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result4.metadata.get('tool_call_status')}")
    print(f"  Tool call adjustment: {result4.metadata.get('tool_call_adjustment')}")
    print()

    # Test case 5: Invalid tool call tags (NEW: -0.5 penalty)
    action5 = """
    <tool_call>
    {"name": "search"}
    No closing tag!
    The answer is **42**.
    """
    reward_input5 = RewardInput(task_info=task_info1, action=action5)
    result5 = reward_fn(reward_input5)
    print(f"Test 5 - Correct answer with invalid tool call tags (NEW: -0.5 penalty):")
    print(f"  Reward: {result5.reward} (Expected: 0.5, 1.0 - 0.5 penalty)")
    print(f"  Is correct: {result5.is_correct}")
    print(f"  Tool call count: {result5.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result5.metadata.get('tool_call_status')}")
    print(f"  Tool call parsing valid: {result5.metadata.get('tool_call_parsing_valid')}")
    print(f"  Tool call adjustment: {result5.metadata.get('tool_call_adjustment')}")
    print()

    # Test case 6: Incorrect answer with multiple tool calls (NEW: -0.5 penalty always)
    action6 = """
    <tool_call>{"name": "search", "arguments": {"query": "first"}}</tool_call>
    <tool_call>{"name": "search", "arguments": {"query": "second"}}</tool_call>
    The answer is wrong.
    """
    reward_input6 = RewardInput(task_info=task_info1, action=action6)
    result6 = reward_fn(reward_input6)
    print(f"Test 6 - Incorrect answer with multiple tool calls (NEW: -0.5 penalty):")
    print(f"  Reward: {result6.reward} (Expected: -0.5, 0.0 - 0.5 penalty)")
    print(f"  Is correct: {result6.is_correct}")
    print(f"  Tool call count: {result6.metadata.get('tool_call_count')}")
    print(f"  Tool call status: {result6.metadata.get('tool_call_status')}")
    print(f"  Tool call adjustment: {result6.metadata.get('tool_call_adjustment')}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tool Call Parsing")
    print("=" * 60)
    test_tool_call_parsing()

    print("\n" + "=" * 60)
    print("Testing Reward Calculation with Tool Call Bonus")
    print("=" * 60)
    test_reward_calculation()
    print("\nAll tests completed!")
