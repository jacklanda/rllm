# Tool Call Bonus Reward Implementation

## Summary

Implemented a "tool call bonus" feature in the search reward system to encourage models to use tools appropriately while avoiding reward hacking. The system rewards correct tool usage and penalizes repeated calls or invalid tag generation.

## Reward Strategy

The implementation follows these rules:

1. **Single tool call + Correct answer (reward > 0)** → **+0.5 bonus**
   - Encourages tool usage only when it leads to correct answers
   - Prevents reward hacking (model just calling tools without answering correctly)

2. **Single tool call + Incorrect answer (reward ≤ 0)** → **No adjustment**
   - No bonus given to avoid reward hacking
   - Model must answer correctly to get the bonus

3. **Multiple tool calls (≥ 2)** → **-0.5 penalty** (regardless of answer correctness)
   - Always penalized to suppress repeated calling behavior
   - Applies even if the answer is correct

4. **Invalid/malformed tool call tags** → **-0.5 penalty**
   - Penalizes unparseable or mismatched tags
   - Encourages proper tag formatting

5. **No tool call** → **No adjustment**
   - Neutral, neither bonus nor penalty

## Changes Made

### Modified File: `rllm/rewards/search_reward.py`

#### 1. Updated Method: `parse_tool_calls()`
- **Location**: Lines 13-43
- **Changes**: Now returns 3 values instead of 2: `(count, matches, is_valid)`
- **Purpose**: Parses and validates `<tool_call>...</tool_call>` patterns in model responses
- **Validation**:
  - Ensures tags are properly paired (same number of opening/closing tags)
  - Uses non-greedy matching to avoid cross-pattern matching
  - Returns `is_valid=False` for mismatched tags
  - Returns count and contents of valid tool calls

#### 2. Updated Method: `__call__()`
- **Location**: Lines 265-327
- **Changes**:
  - Added tool call parsing with validity check (line 274)
  - Implemented new reward adjustment logic (lines 289-316):
    - **Invalid tags**: `-0.5` penalty
    - **Multiple calls (≥2)**: `-0.5` penalty (always)
    - **Single call + reward > 0**: `+0.5` bonus
    - **Single call + reward ≤ 0**: No adjustment (avoid reward hacking)
    - **No call**: No adjustment
  - Enhanced metadata with tool call information:
    - `tool_call_count`: Number of valid tool calls detected
    - `tool_call_status`: Status indicator
    - `tool_call_adjustment`: The actual reward adjustment applied
    - `tool_call_parsing_valid`: Whether parsing succeeded

## Configuration

The tool call bonus/penalty is controlled by the `toolcall_bonus` parameter in `RewardConfig`:
- **Default value**: 0.5
- **Location**: `rllm/rewards/reward_types.py:31`
- **Usage**: Can be configured via the training config

## Reward Calculation Examples

### Example 1: Correct Answer with Single Tool Call ✅ BONUS
```
Base reward: 1.0 (correct)
Tool call bonus: +0.5 (single call + reward > 0)
Final reward: 1.5
```

### Example 2: Incorrect Answer with Single Tool Call ⚠️ NO BONUS
```
Base reward: 0.0 (incorrect)
Tool call bonus: 0.0 (avoid reward hacking)
Final reward: 0.0
```

### Example 3: Correct Answer with Multiple Tool Calls ❌ PENALTY
```
Base reward: 1.0 (correct)
Tool call penalty: -0.5 (multiple calls always penalized)
Final reward: 0.5
```

### Example 4: Incorrect Answer with Multiple Tool Calls ❌ PENALTY
```
Base reward: 0.0 (incorrect)
Tool call penalty: -0.5 (multiple calls always penalized)
Final reward: -0.5
```

### Example 5: Correct Answer with Invalid Tags ❌ PENALTY
```
Base reward: 1.0 (correct)
Tool call penalty: -0.5 (invalid tag parsing)
Final reward: 0.5
```

### Example 6: Correct Answer without Tool Call (Neutral)
```
Base reward: 1.0 (correct)
Tool call adjustment: 0.0 (no tool call)
Final reward: 1.0
```

## Testing

A comprehensive test suite has been created at `test_tool_call_reward.py` that verifies:
1. ✅ Single tool call + correct answer → +0.5 bonus
2. ✅ Single tool call + incorrect answer → 0 adjustment (no reward hacking)
3. ✅ Multiple tool calls + correct answer → -0.5 penalty
4. ✅ Multiple tool calls + incorrect answer → -0.5 penalty
5. ✅ Invalid/mismatched tags → -0.5 penalty
6. ✅ No tool call → 0 adjustment
7. ✅ Proper tag parsing and validation

All tests pass successfully.

## Implementation Details

### Tool Call Pattern Matching
- **Pattern**: `<tool_call>(.*?)</tool_call>` (non-greedy, DOTALL flag)
- **Validation**: Counts opening/closing tags to ensure proper pairing
- **Invalid cases**: Returns `is_valid=False` for mismatched tags or parsing failures

### Reward Adjustment Formula
```python
if not is_valid_parsing:
    adjustment = -toolcall_bonus  # Invalid tags penalty
elif tool_call_count >= 2:
    adjustment = -toolcall_bonus  # Multiple calls penalty (always)
elif tool_call_count == 1 and reward > 0:
    adjustment = +toolcall_bonus  # Single call bonus (only if correct)
elif tool_call_count == 1 and reward <= 0:
    adjustment = 0  # No bonus for incorrect answer (avoid reward hacking)
else:
    adjustment = 0  # No tool call
```

### Metadata Tracking
The implementation adds detailed metadata for analysis and debugging:
- `tool_call_count`: Integer count of valid tool calls
- `tool_call_status`: String status with values:
  - `"single_call_bonus"`: Got bonus for single call + correct answer
  - `"single_call_no_bonus"`: Single call but no bonus (incorrect answer)
  - `"multiple_calls_penalty"`: Penalized for multiple calls
  - `"invalid_tags_penalty"`: Penalized for invalid tag parsing
  - `"no_tool_call"`: No tool call detected
- `tool_call_adjustment`: Float value of the reward adjustment applied
- `tool_call_parsing_valid`: Boolean indicating if parsing succeeded

## Benefits

1. **Encourages proper tool usage**: Models get bonus rewards for using search tools correctly
2. **Prevents reward hacking**: No bonus unless the answer is correct (reward > 0)
3. **Prevents spam**: Multiple tool calls in a single response are always penalized
4. **Enforces quality**: Invalid tag generation is penalized
5. **Maintains correctness priority**: Bonus is additive to base correctness reward
6. **Configurable**: Easy to adjust bonus/penalty via config
7. **Observable**: Full metadata tracking for analysis

## Integration

The feature is automatically active when using `search_reward_fn` in training:
- No changes needed to training scripts
- Uses existing `toolcall_bonus` config parameter
- Backward compatible (defaults to 0.5)

To adjust the bonus/penalty value, modify the config:
```yaml
algorithm:
  reward_config:
    toolcall_bonus: 0.5  # Adjust this value
```

## Code Reference

The implementation can be found in:
- Main implementation: `rllm/rewards/search_reward.py:13-43, 265-327`
- Configuration: `rllm/rewards/reward_types.py:31`
- Test suite: `test_tool_call_reward.py`
