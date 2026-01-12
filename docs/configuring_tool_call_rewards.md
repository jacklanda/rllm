# Configuring Tool Call Bonus Rewards

## Overview

Tool call bonus rewards are **enabled by default** with a bonus/penalty value of `0.5`. The system automatically tracks and rewards/penalizes tool usage behavior during training.

## How It Works

### Tool Call Reward Logic

The reward function applies the following rules:

| Scenario | Adjustment | Reason |
|----------|-----------|--------|
| **Single tool call + correct answer** | `+0.5` | ✅ Encourages effective tool use |
| **Single tool call + wrong answer** | `0` | ⚪ No reward hacking |
| **No tool call** | `0` | ⚪ Neutral |
| **Multiple tool calls (≥2)** | `-0.5` | ❌ Discourages repeated calls |
| **Invalid/malformed tags** | `-0.5` | ❌ Penalizes parse errors |

### Reward Components Logged to Wandb

Each reward component is logged separately for analysis:

```
traj/rewards/pass@1_mean              # F1 score (main accuracy)
traj/rewards/base_reward_mean          # Base reward before adjustments
traj/rewards/tool_call_mean            # Tool call bonus/penalty
traj/rewards/tool_call_count_mean      # Average tool calls per trajectory
traj/rewards/tool_call_status_mean     # Numeric status indicator
traj/rewards/exact_match_mean          # Exact match rate
traj/rewards/repetition_penalty_mean   # Repetition penalty (if enabled)
```

## Configuration Options

### 1. Via Config File (Recommended)

Edit `rllm/trainer/config/agent_ppo_trainer.yaml`:

```yaml
reward:
  # Tool call configuration
  toolcall_bonus: 0.5                    # Change to 0.3, 0.7, etc.

  # Repetition penalty (NEW!)
  apply_repetition_penalty: true         # Enable repetition detection
  repetition_penalty_weight: 0.5         # Weight for penalty
  repetition_max_n: 4                    # N-gram size (1-4)

  # Base rewards
  correct_reward: 1.0
  incorrect_reward: 0.0
```

### 2. Via Command Line Override

Override config from your training script:

```bash
./experiments/search/train_search_agent.sh \
    reward.toolcall_bonus=0.7 \
    reward.apply_repetition_penalty=True \
    reward.repetition_penalty_weight=0.3
```

### 3. Disable Tool Call Bonus

To disable tool call bonuses entirely:

```bash
# Set bonus to 0
reward.toolcall_bonus=0.0
```

## Example Configurations

### Aggressive Tool Use Encouragement
```yaml
reward:
  toolcall_bonus: 0.8              # High bonus
  apply_repetition_penalty: true
  repetition_penalty_weight: 0.7   # Strong repetition penalty
```

### Conservative (Minimal Intervention)
```yaml
reward:
  toolcall_bonus: 0.2              # Small bonus
  apply_repetition_penalty: false
```

### Quality-Focused (Penalize Bad Behavior)
```yaml
reward:
  toolcall_bonus: 0.5
  apply_repetition_penalty: true
  repetition_penalty_weight: 1.0   # Maximum repetition penalty
  repetition_max_n: 5              # Detect longer repetitions
```

## Monitoring in Wandb

Look for these metrics to track tool call behavior:

### Key Metrics to Watch

1. **`traj/rewards/tool_call_mean`**: Average tool call adjustment
   - Positive → Model getting bonuses (good tool use)
   - Negative → Model getting penalties (multiple/invalid calls)

2. **`traj/rewards/tool_call_count_mean`**: Average calls per trajectory
   - Should stabilize around 1-2 for good performance
   - High values (>3) indicate over-use

3. **`traj/rewards/pass@1_mean`**: Overall accuracy
   - Main metric - should increase with good tool use

4. **`traj/rewards/repetition_penalty_mean`**: Repetition detection
   - Negative values indicate repetitive text
   - Should approach 0 as model improves

## Troubleshooting

### Model not using tools?
- Increase `toolcall_bonus` (try 0.7-1.0)
- Check `traj/rewards/tool_call_count_mean` is near 0
- Verify tool definitions in system prompt

### Model making too many tool calls?
- Current config already penalizes this (≥2 calls → -0.5)
- Consider reducing `toolcall_bonus` to 0.3
- Check `traj/rewards/tool_call_status_mean` for penalty rate

### Repetitive responses?
- Enable `apply_repetition_penalty: true`
- Increase `repetition_penalty_weight` (start with 0.5)
- Monitor `traj/rewards/repetition_penalty_mean`

## Implementation Details

### Code Locations

- **Reward computation**: `rllm/rewards/search_reward.py:337-416`
- **Metric extraction**: `rllm/engine/agent_execution_engine.py:396-434`
- **Logging**: `rllm/trainer/verl/agent_ppo_trainer.py:629-643`
- **Configuration**: `rllm/trainer/config/agent_ppo_trainer.yaml:113-124`

### Factory Function

The reward function is created using:

```python
from rllm.rewards.reward_fn import create_search_reward_fn

reward_fn = create_search_reward_fn(
    toolcall_bonus=0.5,
    apply_repetition_penalty=False,
    repetition_penalty_weight=0.5,
    repetition_max_n=4,
    correct_reward=1.0,
    incorrect_reward=0.0,
)
```

## Summary

✅ **Tool call bonuses are enabled by default with value 0.5**
✅ **All reward components are logged separately to wandb**
✅ **Easily configurable via YAML or command line**
✅ **Repetition penalty now supported**

No additional setup needed - just adjust the config values to tune behavior!
