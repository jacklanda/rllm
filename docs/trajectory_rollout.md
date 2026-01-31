# 轨迹 Rollout (轨迹生成) 流程

本文档详细介绍了 `rllm/engine/agent_execution_engine.py` 中实现的轨迹 rollout 流程。`AgentExecutionEngine` 负责管理智能体（Agent）与环境（Environment）之间的交互循环、执行动作、观察状态，并收集用于训练或评估的轨迹数据。

## 概述

`AgentExecutionEngine` 编排多个智能体的并行执行。它主要处理：
- 智能体与环境交互的异步执行。
- 与不同 Rollout 后端（OpenAI, Verl, Tinker）的集成。
- 针对模型输出的健壮错误处理和重试机制。
- 以多种格式（Text, Token, Conversation, Step）收集轨迹数据。

## 核心组件

### AgentExecutionEngine（智能体执行引擎）

主类初始化包含以下内容：
- **智能体与环境**：用于实例化并行实例的类和参数。
- **Rollout Engine**：用于 LLM 推理的后端（例如 `OpenAIEngine`, `VerlEngine`）。
- **配置**：`max_steps`（最大步数）、`max_prompt_length`（最大提示词长度）、`timeout`（超时时间）等参数。

### Rollout Engines（推引起擎）

该引擎支持可插拔的模型推理后端：
- **OpenAI**：使用 OpenAI 兼容的 API。
- **Verl**：与 Verl 强化学习框架集成。
- **Tinker**：支持自定义引擎。

## Rollout 工作流

核心逻辑位于 `run_agent_trajectory_async` 方法中，它从头到尾执行单条轨迹。

### 1. 初始化
- 通过 `env.reset()` 重置环境。
- 重置智能体并根据初始观察值更新智能体状态。
- 计算初始提示词 Token 长度，并检查是否超过 `max_prompt_length`。

### 2. 交互循环
引擎在 `max_steps` 限制内进行迭代：

#### a. 提示词构建 (Prompt Construction)
- 将智能体的对话历史转换为提示词。
- 如果启用了 `enforce_max_prompt_length`，则验证提示词长度。

#### b. 模型推理与验证 (Model Inference & Validation)
- 向模型请求响应。
- **重试机制**：引擎针对无效输出实现了健壮的重试循环（默认 32 次）：
    - **无效输出判定**：既没有工具调用（Tool Call），也没有 `\boxed{}` 形式的答案（除非是因为长度原因结束）。
    - 如果输出无效，则进行重试。
    - 如果重试次数耗尽，则以 `ABNORMAL_PARSE_ERROR`（异常解析错误）终止。

#### c. 动作执行 (Action Execution)
- 将模型响应解析为 `Action`（动作）。
- 在环境中执行动作 (`env.step(action)`)。
- **超时处理**：监控环境执行。如果超过 `trajectory_timeout`，轨迹将被丢弃 (`ENV_TIMEOUT`)。

#### d. 状态更新 (State Update)
- 根据新的 `observation`（观察）、`reward`（奖励）和 `done`（终止信号）更新智能体。
- 追踪中间奖励。

#### e. 终止检查 (Termination Checks)
- **上下文截断**：如果响应 Token 超过 `max_response_length`，轨迹以 `TRUNCATION` 终止。
- **环境结束**：如果 `done` 为 True，循环结束 (`ENV_DONE`)。
- **全局超时**：如果总执行时间超过限制，轨迹以 `TIMEOUT` 终止。
- **工具爆发 (Tool Burst)**：如果单步生成超过 10 个工具调用，轨迹以 `ABNORMAL_TOOL_BURST` 终止。
- **重复查询**：检测并终止重复的搜索查询 (`ABNORMAL_REPEATED_QUERY`)。

### 3. 后处理
循环结束后：

- **ReAct 结构验证**：
    - 要求至少 5 个步骤（意味着一个有意义的 ReAct 追踪所需的最小长度）。
    - 检查最后一步是否包含工具调用但没有最终答案。
    - 如果验证失败，抛出 `InvalidReactStructureError`（这将触发轨迹级别的重试）。

- **最终奖励计算**：
    - 如果环境支持 `compute_final_reward`，则调用该方法。
    - 异常终止（解析错误、工具爆发等）通常导致 0 奖励。

- **统计**：计算轨迹的蒙特卡洛（MC）回报。

### 4. 输出生成
根据指定的模式返回轨迹：
- **Text**：返回高层级的 `Trajectory` 对象。
- **Token**：返回原始 Token 和 Mask，适用于 RL 训练（如 PPO）。包含详细指标（F1, 精确匹配, 时间统计）。
- **Conversation**：返回消息字典列表。
- **Step**：返回包含奖励和回报的逐步分解数据。

## 并行执行

`trajectory_generator` 方法管理并发执行：
- 使用 `asyncio.Semaphore` 将并发量限制在 `n_parallel_agents`。
- 使用 `ThreadPoolExecutor` 处理阻塞的环境操作（reset, step, close）。
- 实现轨迹级别的重试机制：
    - 针对 `InvalidReactStructureError` 最多重试 8 次。
    - 针对其他异常重试 `retry_limit` 次。

## 失败模式与终止原因

| 原因 (Reason) | 描述 | 奖励 (Reward) |
| :--- | :--- | :--- |
| `ENV_DONE` | 成功到达终止状态。 | 正常计算 |
| `MAX_STEPS` | 超过最大允许步数。 | 强制 0.0 |
| `TRUNCATION` | 超过上下文窗口（响应长度）。 | 0.0 (被 Mask) |
| `TIMEOUT` | 超过全局执行时间限制。 | 当前累计值 |
| `ENV_TIMEOUT` | 单步环境执行耗时过长。 | 直接丢弃 |
| `ABNORMAL_PARSE_ERROR` | 模型多次重试后仍无法产生有效输出。 | 0.0 |
| `ABNORMAL_TOOL_BURST` | 单步内工具调用过多。 | 0.0 |
| `ABNORMAL_REPEATED_QUERY` | 检测到重复的搜索查询。 | 0.0 |
| `INVALID_REACT_STRUCTURE` | ReAct 轨迹格式错误（太短或未完成）。 | 直接丢弃 |
