# GDPO Implementation Summary

## Overview
This document summarizes the changes made to implement GDPO (Group-based Direct Preference Optimization) support in the agent PPO trainer, adapting it to work with the reward structure from `search_reward.py`.

## Changes Made

### 1. Updated `ray_trainer.py`

#### Function Signature Update
- **File**: `rllm/trainer/verl/ray_trainer.py`
- **Function**: `compute_advantage`
- **Change**: Added missing parameters `norm_adv_by_std_in_grpo` and `config` to match the call signature in `agent_ppo_trainer.py`

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, norm_adv_by_std_in_grpo=True, config=None):
```

#### GDPO Advantage Computation
- **Updated**: The `gdpo` branch in `compute_advantage` to work with agent reward structure
- **Old approach**: Used `token_level_scores_correctness` and `token_level_scores_format`
- **New approach**: Uses `token_level_rewards_base` (correctness/answer quality) and `token_level_rewards_bonus` (tool_call + step_bonus)
- **Fallback**: If component rewards are not available, uses full reward as base and zero for bonus

```python
elif adv_estimator == "gdpo":
    ## Handle two reward components: base (correctness) and bonus (tool_call + step_bonus)
    if "token_level_rewards_base" in data.batch and "token_level_rewards_bonus" in data.batch:
        token_level_rewards_base = data.batch["token_level_rewards_base"]
        token_level_rewards_bonus = data.batch["token_level_rewards_bonus"]
    else:
        # Fallback: use the full reward as base, and zero for bonus
        token_level_rewards_base = data.batch["token_level_rewards"]
        token_level_rewards_bonus = torch.zeros_like(token_level_rewards_base)

    # Normalize each component separately using GRPO
    base_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(...)
    bonus_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(...)

    # Combine and whiten
    new_advantage = base_normalized_score + bonus_normalized_score
    advantages = masked_whiten(new_advantage, response_mask) * response_mask
```

#### Metrics Update
- **Updated**: `compute_data_metrics` function to handle optional reward components
- **Added**: Support for new reward components (`token_level_scores_base`, `token_level_scores_bonus`)
- **Maintained**: Backward compatibility with old reward structure (`token_level_scores_format`, `token_level_scores_correctness`, `token_level_scores_length`)

### 2. Updated `single_turn_env.py`

#### Reward Metadata Storage
- **File**: `rllm/environments/base/single_turn_env.py`
- **Function**: `get_reward_and_next_obs`
- **Change**: Store reward metadata in the info dict for later use in GDPO

```python
def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
    reward_output = self.reward_fn(task_info=task, action=action)

    # Store reward metadata in the info dict for later use in GDPO
    info = {"reward_metadata": reward_output.metadata} if hasattr(reward_output, "metadata") else {}

    return reward_output.reward, info
```

### 3. Updated `agent_ppo_trainer.py`

#### Trajectory Transformation (`_transform_agent_trajectories`)
- **Added**: Lists to store base rewards and bonus rewards separately
- **Extraction**: Extract reward components from trajectory metadata:
  - `base_reward`: Main reward based on answer correctness
  - `tool_call_reward`: Bonus/penalty for tool usage
  - `step_bonus`: Bonus for using more steps
- **Storage**: Store these components in separate tensors in the batch

```python
# Extract reward components from metadata for GDPO
reward_metadata = traj.get("reward_metadata", {})
base_reward = reward_metadata.get("base_reward", traj["trajectory_reward"])
tool_call_reward = reward_metadata.get("tool_call_reward", 0.0)
step_bonus = reward_metadata.get("step_bonus", 0.0)

# Store base reward and bonus rewards separately
traj_base_rewards.append(base_reward)
traj_bonus_rewards.append(tool_call_reward + step_bonus)
```

- **Added to tensor_batch**:
  - `token_level_scores_base`: Base reward tensor
  - `token_level_scores_bonus`: Bonus reward tensor

#### Step Transformation (`_transform_agent_steps`)
- **Similar changes** as trajectory transformation for stepwise advantage mode
- **Added**: Base and bonus reward tracking for each step
- **Storage**: Store component rewards in the tensor batch

#### Advantage Computation Setup
- **Added**: Logic to set up component rewards for GDPO before advantage computation

```python
# For GDPO, also set up the component rewards
if self.config.algorithm.adv_estimator == "gdpo":
    if "token_level_scores_base" in batch.batch:
        batch.batch["token_level_rewards_base"] = batch.batch["token_level_scores_base"]
    if "token_level_scores_bonus" in batch.batch:
        batch.batch["token_level_rewards_bonus"] = batch.batch["token_level_scores_bonus"]
```

### 4. Updated `agent_execution_engine.py`

#### Reward Metadata Extraction and Storage
- **File**: `rllm/engine/agent_execution_engine.py`
- **Change**: Extract and store full reward metadata from the last step
- **Added**: `reward_metadata` variable to store the complete metadata dict
- **Included**: Add `reward_metadata` to both Token and Step mode results

```python
# Extract reward components from the last step's metadata
reward_metrics = {}
reward_metadata = {}  # Store full metadata for GDPO
if trajectory.steps:
    last_step = trajectory.steps[-1]
    if "metadata" in last_step.info:
        metadata = last_step.info["metadata"]
        reward_metadata = metadata  # Store full metadata
        ...

token_result = {
    ...
    "reward_metadata": reward_metadata,  # Add reward metadata for GDPO
    ...
}
```

## Reward Structure

### From `search_reward.py`
The reward function returns a `RewardOutput` object with:
- `reward`: Final combined reward
- `is_correct`: Boolean indicating correctness
- `metadata`: Dict containing:
  - `base_reward`: Main reward based on answer quality (F1 score or exact match)
  - `tool_call_reward`: Bonus/penalty for tool usage (currently set to 0)
  - `step_bonus`: Bonus for using more steps (if enabled)
  - `repetition_penalty_reward`: Penalty for repetition (currently set to 0)

### GDPO Decomposition
For GDPO, we split the reward into two components:
1. **Base Reward** (`token_level_scores_base`): The correctness/answer quality reward
2. **Bonus Reward** (`token_level_scores_bonus`): Sum of tool_call_reward and step_bonus

Each component is normalized separately using GRPO's outcome advantage computation, then combined and whitened.

## Usage

To use GDPO with the agent PPO trainer:

1. Set `algorithm.adv_estimator: "gdpo"` in your config
2. Ensure your reward function returns a `RewardOutput` with metadata containing `base_reward`, `tool_call_reward`, and `step_bonus`
3. The trainer will automatically extract these components and use them for GDPO advantage computation

## Backward Compatibility

All changes maintain backward compatibility:
- If reward components are not available, the system falls back to using the full reward
- Old reward structures (`token_level_scores_format`, etc.) are still supported
- Metrics are only logged if the corresponding reward components exist

## Testing Recommendations

1. Test with `adv_estimator: "gdpo"` to ensure GDPO works correctly
2. Test with `adv_estimator: "grpo"` to ensure backward compatibility
3. Verify that reward components are correctly extracted and logged in metrics
4. Check that the advantage computation produces reasonable values
