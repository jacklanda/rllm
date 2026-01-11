# Trajectory-Level Filtering for SimpleTIR

This document explains how to use trajectory-level filtering to reproduce the SimpleTIR paper's approach to handling invalid trajectories in PPO training.

## Overview

SimpleTIR requires masking out/excluding entire trajectories that contain "invalid turns" (e.g., incomplete code blocks or failed tool calls) from the gradient update to prevent training instability. This implementation provides granular filtering at the individual trajectory level within a batch, without breaking advantage normalization or PPO/GRPO batch structure.

## Key Features

- **Trajectory-level granularity**: Filter individual trajectories within episodes
- **Multiple validation criteria**: Detect incomplete code blocks, failed tool calls, low rewards, excessive steps
- **Maintains batch structure**: Properly handles advantage normalization and PPO updates
- **Logging**: Detailed logs show which trajectories are filtered and why
- **Compatible with existing filtering**: Works alongside compact_filtering for comprehensive filtering

## How It Works

### Architecture

The filtering is implemented in `rllm/engine/agent_workflow_engine.py`:

1. **Validation Functions** (lines 24-119):
   - `_has_incomplete_code_blocks()`: Detects unclosed markdown code blocks
   - `_has_failed_tool_calls()`: Checks for errors in observations and info dicts
   - `_validate_trajectory()`: Applies all configured validation criteria

2. **Filtering Logic** (lines 372-376 in `transform_results_for_verl()`):
   - Applied during trajectory processing, before tokenization
   - Invalid trajectories are skipped (not included in batch)
   - Maintains proper repeat_counts for batch structure

### Validation Criteria

#### 1. Incomplete Code Blocks
Detects markdown code blocks that are not properly closed:
```python
# Valid: "Here's code: ```python\nprint('hello')\n```"
# Invalid: "Here's code: ```python\nprint('hello')"  # Missing closing ```
```

#### 2. Failed Tool Calls
Detects errors in step observations or info dicts:
```python
# Checked in step.observation for keywords: 'error', 'exception', 'failed', 'failure', 'traceback'
# Checked in step.info for has_error flag
```

#### 3. Reward Threshold
Filters trajectories below a specified reward:
```yaml
trajectory_filtering:
  min_reward: 0.5  # Only keep trajectories with reward >= 0.5
```

#### 4. Max Steps
Filters trajectories exceeding a maximum number of steps:
```yaml
trajectory_filtering:
  max_steps: 20  # Filter trajectories with more than 20 steps
```

## Configuration

### Basic Configuration

Add to your training config YAML:

```yaml
rllm:
  trajectory_filtering:
    enable: True
    filter_incomplete_code_blocks: True
    filter_failed_tool_calls: True
    min_reward: null  # or set to a float value
    max_steps: null   # or set to an integer
```

### SimpleTIR Reproduction Config

For strict SimpleTIR reproduction:

```yaml
rllm:
  trajectory_filtering:
    enable: True
    filter_incomplete_code_blocks: True
    filter_failed_tool_calls: True
    min_reward: null
    max_steps: null
```

### Combined with Compact Filtering

You can use both filtering mechanisms together:

```yaml
rllm:
  # Episode-level filtering based on termination reasons
  compact_filtering:
    enable: True
    mask_max_prompt_length_exceeded: True
    mask_max_response_length_exceeded: True
    mask_max_turns_exceeded: True
    mask_timeout: True
    mask_error: True

  # Trajectory-level filtering based on content validation
  trajectory_filtering:
    enable: True
    filter_incomplete_code_blocks: True
    filter_failed_tool_calls: True
    min_reward: null
    max_steps: null
```

## Usage Example

### Training Script

```python
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.workflows.workflow import Workflow

# Your workflow implementation
class MyWorkflow(Workflow):
    async def run(self, task, uid, **kwargs):
        # Your workflow logic here
        # Return Episode with trajectories
        pass

# Initialize trainer with trajectory filtering enabled
trainer = AgentWorkflowPPOTrainer(
    config=config,  # Config with trajectory_filtering enabled
    workflow_cls=MyWorkflow,
    # ... other args
)

# Training will automatically filter invalid trajectories
trainer.fit()
```

### Expected Log Output

When filtering is active, you'll see logs like:

```
INFO - Filtering out trajectory 1234567890_solver: incomplete_code_blocks
INFO - Filtering out trajectory 1234567891_judge: failed_tool_calls
INFO - Filtering out trajectory 1234567892_solver: reward_below_threshold_0.5
INFO - Filtering out trajectory 1234567893_judge: exceeded_max_steps_20
```

## Implementation Details

### Batch Structure Maintenance

The filtering maintains proper PPO batch structure:

1. **Trajectory skipping**: Invalid trajectories are skipped during batch construction
2. **Repeat counts**: Episode repeat counts automatically adjust based on valid trajectories
3. **Advantage normalization**: Only valid trajectories are included in advantage computation
4. **Gradient updates**: Only valid trajectories contribute to gradients

### When Filtering Happens

```
Episode Generation → Transform to DataProto → [FILTERING HAPPENS HERE] → Tokenization → Batch Construction → PPO Update
```

Filtering occurs early in `transform_results_for_verl()`, before trajectories are tokenized and added to the batch.

### Empty Episodes

If all trajectories in an episode are filtered:
- The episode's `repeat_counts` is set to 0
- The episode is effectively removed from the batch
- Logged as: "Episode {id} has no valid trajectories, dropping it from the batch"

## Custom Validation

You can extend the validation logic by modifying `_validate_trajectory()` in `agent_workflow_engine.py`:

```python
def _validate_trajectory(trajectory, config):
    # ... existing validation ...

    # Add custom validation
    if config.rllm.trajectory_filtering.custom_check:
        if not my_custom_check(trajectory):
            return False, "custom_validation_failed"

    return True, ""
```

## Performance Considerations

- **Minimal overhead**: Validation is simple string/list checks
- **Early filtering**: Happens before expensive tokenization
- **Parallel processing**: Trajectory validation is independent
- **Logging impact**: Set logging level to WARNING if filter logs are too verbose

## Troubleshooting

### Too many trajectories filtered

Check your validation criteria:
```bash
# View filter statistics in logs
grep "Filtering out trajectory" training.log | wc -l
grep "Filtering out trajectory.*incomplete_code_blocks" training.log | wc -l
grep "Filtering out trajectory.*failed_tool_calls" training.log | wc -l
```

Adjust thresholds if needed:
```yaml
trajectory_filtering:
  filter_incomplete_code_blocks: False  # Disable if too strict
  min_reward: -1.0  # Lower threshold if filtering too much
```

### Not enough trajectories filtered

Verify configuration is enabled:
```yaml
trajectory_filtering:
  enable: True  # Must be True
```

Check logs for filtering messages - if none appear, filtering may not be active.

## References

- SimpleTIR Paper: [Link to paper if available]
- Code: `rllm/engine/agent_workflow_engine.py` (lines 24-119, 372-376)
- Config: `rllm/trainer/config/agent_ppo_trainer.yaml` (lines 62-69)

## See Also

- [Example Configuration](./trajectory_filtering_example.yaml)
- [Workflow Documentation](../docs/core-concepts/workflow-engine.md)
- [Compact Filtering](../docs/compact_filtering.md)
