# Search Agent Trajectory Filtering (SimpleTIR Implementation)

This document explains the implementation of trajectory-level filtering for multi-turn search agents based on SimpleTIR Section 5.4: Abnormal Trajectory Handling.

## Overview

During RL training of multi-turn search agents, there is a certain probability of rolling out abnormal trajectories. These abnormal trajectories typically have more extreme PPO importance ratios and require special handling to prevent training instability.

This implementation categorizes anomalies into **model-induced** and **environment-induced**, and applies unified, reproducible handling rules.

## Abnormal Trajectory Categories

### 1. Discard (5.4.2): Environment-Induced Errors + Critical Parse Exceptions

**Behavior**: Completely discard from batch

**Triggers**:
- **Tool call parse exceptions** (NEW): "Error parsing tool call", "Failed to parse tool call", etc.
- Search/retrieval timeouts
- Connection errors
- Service unavailable errors
- Network errors

**Rationale**:
- **Parse exceptions**: These represent complete failures where the model output cannot be parsed into tool calls at all. Including them would contaminate the training signal with completely invalid data.
- **Environment errors**: These errors are caused by environment instability, not model behavior. Including them in training would add noise unrelated to model policy.

**Implementation**: Trajectories are detected and skipped entirely in `transform_results_for_verl()` before tokenization.

**Detection locations**:
- `step.observation` containing error messages
- `step.model_response` containing error messages
- `step.info['tool_call_parse_error']` or `step.info['parse_error']` flags
- `step.info['error']` or `step.info['error_message']` containing error keywords
- `chat_completions` messages containing error keywords

### 2. Zero Reward (5.4.1, 5.4.3): Model-Induced Anomalies

**Behavior**: Keep in batch for advantage computation AND gradient updates, but assign 0 reward

**Triggers**:
- Tool call count in single turn exceeds limit (default: >10)
- Tool parse errors (unclosed `<think>` tags, malformed JSON)
- Repeated queries (identical search query to previous queries)
- Exceeding search turn limit (default: >20 turns)

**Rationale**: These are model behaviors we want to suppress. Setting reward to 0 provides a strong signal against these behaviors while maintaining batch structure.

**Implementation**: Trajectory reward and all step rewards are set to 0, but trajectory remains in batch.

### 3. No Gradient (5.4.4): Token Limit Exceeded

**Behavior**: Keep in batch for advantage computation, but exclude from gradient updates

**Triggers**:
- Total token count exceeds maximum (default: >8192 tokens)

**Rationale**: Long trajectories provide useful signal for advantage normalization but have higher probability of anomalies. Using them for advantages but not gradients reduces contamination while maintaining statistical validity.

**Implementation**: Trajectories are marked with `no_grad=True` flag in the batch. The trainer can use this flag to filter samples before loss computation.

## Implementation Details

### Code Structure

**File**: `rllm/engine/agent_workflow_engine.py`

**Key Functions**:

1. **_count_tool_calls_in_message()** (lines 24-46)
   - Counts tool calls in assistant messages
   - Supports multiple patterns: `<tool_call>` tags, function calls, JSON format

2. **_has_tool_parse_error()** (lines 49-86)
   - Detects structural parse errors in model output
   - Detects unclosed tags (`<think>`, `<tool_call>`)
   - Detects malformed JSON parameters
   - Checks for unmatched brackets and quotes

3. **_has_tool_call_parse_exception()** (lines 89-144) **NEW**
   - Detects critical "Error parsing tool call" exception messages
   - Checks in observations, responses, info dicts, and chat_completions
   - Recognizes multiple error message patterns:
     - "error parsing tool call"
     - "failed to parse tool call"
     - "tool call parse error"
     - "cannot parse tool call"
     - "invalid tool call format"
     - "tool call parsing failed"

4. **_extract_search_query()** (lines 147-172)
   - Extracts query from various formats
   - Supports: `query="..."`, `"query": "..."`, `<query>...</query>`

5. **_has_repeated_query()** (lines 175-205)
   - Tracks all queries in trajectory
   - Detects duplicate queries

6. **_has_search_error()** (lines 208-238)
   - Detects environment-induced errors
   - Keywords: timeout, connection error, retrieval error, etc.

7. **_validate_trajectory()** (lines 241-304)
   - Main validation function
   - **Checks parse exceptions first** (critical)
   - Returns action: "keep", "discard", "zero_reward", or "no_grad"

8. **_compute_token_count()** (lines 307-329)
   - Computes total tokens for trajectory
   - Uses model_output when available, estimates otherwise

### Filtering Flow

```
Episode Generation
  ↓
Transform to DataProto
  ↓
For each trajectory:
  ├─ Validate trajectory → action, reason
  ├─ If "discard":
  │   ├─ tool_call_parse_exception → skip (critical error)
  │   └─ search_error_environment → skip (env error)
  ├─ If "zero_reward": set trajectory.reward = 0
  └─ If max_tokens exceeded: mark no_grad = True
  ↓
Process trajectory (tokenize, pad, etc.)
  ↓
Add to batch with metadata
  ↓
Batch contains:
  - Tensors: input_ids, attention_mask, rewards, etc.
  - Non-tensors: is_valid, no_grad, trajectory_ids, etc.
  ↓
Trainer uses no_grad flag to filter before loss computation
```

### Batch Metadata

The `no_grad` flag is added to the batch's non_tensors dict:

```python
non_tensors = {
    "episode_ids": ...,
    "trajectory_ids": ...,
    "step_ids": ...,
    "is_valid": ...,  # From compact_filtering
    "no_grad": ...,   # New: for 5.4.4 handling
    ...
}
```

## Configuration

### Basic Configuration

Add to your training config YAML:

```yaml
rllm:
  trajectory_filtering:
    enable: True
    max_tool_calls_per_turn: 10      # 5.4.1: Filter if >N tool calls
    max_search_turns: 20             # 5.4.3: Filter if >N turns
    max_tokens: 8192                 # 5.4.4: Mark as no_grad if >N tokens
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | False | Enable trajectory filtering |
| `max_tool_calls_per_turn` | int | 10 | Maximum concurrent tool calls in one turn |
| `max_search_turns` | int\|null | null | Maximum search turns before 0 reward |
| `max_tokens` | int\|null | null | Maximum tokens before no_grad marking |

**Automatically detected** (no configuration needed):
- Tool parse errors
- Repeated queries
- Environment search errors

### Example Configurations

**SimpleTIR Reproduction (Strict)**:
```yaml
trajectory_filtering:
  enable: True
  max_tool_calls_per_turn: 10
  max_search_turns: 20
  max_tokens: 8192
```

**Relaxed Filtering**:
```yaml
trajectory_filtering:
  enable: True
  max_tool_calls_per_turn: 20
  max_search_turns: 50
  max_tokens: 16384
```

**Minimal Filtering** (only critical errors):
```yaml
trajectory_filtering:
  enable: True
  max_tool_calls_per_turn: 50
  max_search_turns: null  # Disabled
  max_tokens: null        # Disabled
```

## Usage

### Training Script

```python
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer

# Config with trajectory_filtering enabled
trainer = AgentWorkflowPPOTrainer(
    config=config,
    workflow_cls=MySearchWorkflow,
    ...
)

# Filtering happens automatically during training
trainer.fit()
```

### Expected Log Output

When filtering is active, you'll see detailed logs:

```
INFO - Discarding trajectory 1234567890_agent: tool_call_parse_exception
INFO - Discarding trajectory 1234567891_agent: search_error_environment
INFO - Setting zero reward for trajectory 1234567892_agent: tool_call_limit_exceeded_15
INFO - Setting zero reward for trajectory 1234567893_agent: tool_parse_error
INFO - Setting zero reward for trajectory 1234567894_agent: repeated_query
INFO - Setting zero reward for trajectory 1234567895_agent: max_search_turns_exceeded_20
INFO - Marking trajectory 1234567896_agent as no_grad: token_count=9000 > max_tokens=8192
```

### Monitoring Filtering Statistics

You can track filtering statistics in your logs:

```bash
# Count each filter type
grep "Discarding trajectory" training.log | wc -l
grep "Discarding trajectory.*tool_call_parse_exception" training.log | wc -l
grep "Discarding trajectory.*search_error_environment" training.log | wc -l
grep "Setting zero reward.*tool_call_limit" training.log | wc -l
grep "Setting zero reward.*tool_parse_error" training.log | wc -l
grep "Setting zero reward.*repeated_query" training.log | wc -l
grep "Setting zero reward.*max_search_turns" training.log | wc -l
grep "Marking trajectory.*no_grad" training.log | wc -l
```

## Handling no_grad in Trainer

The trainer should filter samples with `no_grad=True` before computing loss:

```python
# After computing advantages but before loss
no_grad = batch.non_tensor_batch["no_grad"]
grad_idxs = np.where(no_grad == False)[0]
batch_for_loss = batch.select_idxs(grad_idxs)

# Compute loss only on batch_for_loss
loss = compute_ppo_loss(batch_for_loss)
```

**Note**: The current implementation adds the `no_grad` flag to the batch. The trainer integration for filtering before loss computation is left for you to implement based on your specific trainer architecture.

## Comparison with Compact Filtering

| Feature | Compact Filtering | Trajectory Filtering |
|---------|------------------|---------------------|
| **Granularity** | Episode-level | Trajectory-level |
| **Trigger** | Termination reasons | Content validation |
| **Timing** | After batch creation | During batch creation |
| **Filtering** | One type (is_valid) | Three types (discard/zero_reward/no_grad) |
| **Use Case** | System limits exceeded | Abnormal behaviors |

Both can be used together for comprehensive filtering.

## Key Differences from Generic Filtering

The search-specific implementation differs from generic trajectory filtering:

1. **Three-tier filtering**: discard/zero_reward/no_grad vs. simple valid/invalid
2. **Search-specific detection**: Tool calls, queries, parse errors
3. **Automatic detection**: No configuration needed for many anomalies
4. **Token-based handling**: Special treatment for long trajectories
5. **Environment vs. model**: Distinguishes error sources

## Limitations and Considerations

1. **Query extraction patterns**: May not match all query formats. Extend `_extract_search_query()` for custom formats.

2. **Token counting accuracy**: Uses rough estimates when `model_output` unavailable. For precise counting, ensure your workflow populates `model_output`.

3. **No_grad trainer integration**: The `no_grad` flag is provided in the batch, but you must implement filtering in your trainer before loss computation.

4. **Parse error detection**: Heuristic-based. May have false positives/negatives for complex formats.

5. **Performance**: Validation adds minimal overhead (<1ms per trajectory), but logging can be verbose at INFO level.

## Troubleshooting

### Too Many Trajectories Filtered

**Symptom**: Most trajectories are being filtered

**Solutions**:
- Increase limits: `max_tool_calls_per_turn`, `max_search_turns`, `max_tokens`
- Check logs to see which filter is triggering most often
- Verify your workflow is generating valid output formats

### Unexpected Filtering

**Symptom**: Valid-looking trajectories are filtered

**Solutions**:
- Enable DEBUG logging to see exact detection logic
- Check if your query format matches extraction patterns
- Verify tool call format matches expected patterns

### Not Enough Filtering

**Symptom**: Abnormal trajectories passing through

**Solutions**:
- Lower limits for stricter filtering
- Check if your error messages match environment error keywords
- Add custom detection logic in validation functions

## Extending the Implementation

### Adding Custom Query Patterns

Edit `_extract_search_query()`:

```python
# Add your custom pattern
match = re.search(r'YOUR_PATTERN', text)
if match:
    return match.group(1).strip()
```

### Adding Custom Error Detection

Edit `_has_search_error()`:

```python
# Add your custom error keywords
custom_keywords = ['your_error', 'custom_failure']
if any(keyword in obs_str for keyword in custom_keywords):
    return True
```

### Custom Validation Logic

Add at the end of `_validate_trajectory()`:

```python
# Custom validation
if your_custom_condition(trajectory):
    return "zero_reward", "custom_reason"
```

## References

- SimpleTIR Paper: Section 5.4 Abnormal Trajectory Handling
- Implementation: `rllm/engine/agent_workflow_engine.py`
- Config: `rllm/trainer/config/agent_ppo_trainer.yaml`
- Example: `examples/trajectory_filtering_example.yaml`

## See Also

- [Compact Filtering](./compact_filtering.md)
- [Workflow Documentation](./workflow-engine.md)
- [PPO Trainer](./ppo_trainer.md)
