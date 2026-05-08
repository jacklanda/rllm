import asyncio
import json
import math
import os
import uuid


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return str(o)
        try:
            return super().default(o)
        except TypeError:
            return str(o)
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role, WorkerType
from rllm.trainer.verl.ray_trainer import compute_advantage, compute_data_metrics
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine


def _patch_rlhf_dataset_answer_norm():
    import datasets
    from verl.utils.dataset.rl_dataset import RLHFDataset

    original = RLHFDataset._read_files_and_tokenize

    def patched(self):
        dataframes = []
        for parquet_file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            # Normalize top-level answer column
            if "answer" in df.features:
                if not isinstance(df.features["answer"], datasets.Value):
                    df = df.map(lambda x: {"answer": (x["answer"][0] if x["answer"] else "") if isinstance(x["answer"], list) else str(x["answer"] or "")})
                new_features = df.features.copy()
                new_features["answer"] = datasets.Value("string")
                df = df.cast(new_features)
            # Build prompt from input if prompt is null (val benchmark datasets)
            if "prompt" in df.features and isinstance(df.features["prompt"], datasets.Value) and df.features["prompt"].dtype == "null":
                if "input" in df.features:
                    df = df.map(lambda x: {"prompt": [{"role": "user", "content": x["input"]}]})
            # Promote top-level data_source into extra_info before schema alignment
            # so it survives the common-column intersection across datasets
            if "data_source" in df.column_names and "extra_info" in df.column_names:
                import json as _json
                def _inject_data_source(x):
                    ei = x["extra_info"]
                    if isinstance(ei, str):
                        ei = _json.loads(ei) if ei else {}
                    if not isinstance(ei, dict):
                        ei = {}
                    if "data_source" not in ei and x.get("data_source"):
                        ei["data_source"] = x["data_source"]
                    return {"extra_info": ei}
                df = df.map(_inject_data_source)
            # Serialize extra_info struct to JSON string so schemas are compatible across datasets
            if "extra_info" in df.column_names:
                import json as _json
                df = df.map(lambda x: {"extra_info": _json.dumps(x["extra_info"]) if not isinstance(x["extra_info"], str) else x["extra_info"]})
                new_features = df.features.copy()
                new_features["extra_info"] = datasets.Value("string")
                df = df.cast(new_features)
            dataframes.append(df)
        # Align schemas: keep only columns present in all datasets
        common_cols = set(dataframes[0].column_names)
        for df in dataframes[1:]:
            common_cols &= set(df.column_names)
        dataframes = [df.select_columns(list(common_cols)) for df in dataframes]
        self.dataframe = datasets.concatenate_datasets(dataframes)
        # run the rest of the original method after dataframe is set
        import numpy as np
        total = len(self.dataframe)
        print(f"dataset len: {total}")
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    RLHFDataset._read_files_and_tokenize = patched


_patch_rlhf_dataset_answer_norm()


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            eval_trajectory_timeout=self.config.rllm.agent.get("eval_trajectory_timeout", None),
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def _check_docker_connectivity(self):
        """Pre-flight check: verify Docker daemon is reachable before launching trajectories."""
        docker_host = os.environ.get("DOCKER_HOST", "")
        if not docker_host:
            return  # Using local socket, skip remote check

        try:
            import docker
            client = docker.from_env(timeout=10)
            client.ping()
            client.close()
            print(f"Docker health check passed (host={docker_host})")
        except Exception as e:
            raise RuntimeError(
                f"Docker daemon is unreachable at {docker_host}: {e}\n"
                f"Please ensure the Docker daemon is running and accessible. "
                f"You can verify with: DOCKER_API_VERSION=1.44 docker -H {docker_host} info"
            ) from e

        # Pre-seed R2E-Gym's shared Docker client with a large HTTP connection
        # pool. With hundreds of parallel agents, the default pool size of 10
        # saturates and urllib3 logs "Connection pool is full, discarding
        # connection" — dropped connections are opened again, wasting time.
        try:
            from r2egym.agenthub.runtime.docker import DockerRuntime
            n_parallel = int(self.config.rllm.agent.get("engine_args", {}).get("n_parallel_agents", 64) or 64)
            pool_size = max(64, n_parallel + 32)
            if DockerRuntime._shared_docker_client is None:
                with DockerRuntime._client_lock:
                    if DockerRuntime._shared_docker_client is None:
                        DockerRuntime._shared_docker_client = docker.DockerClient(
                            base_url=docker_host,
                            timeout=120,
                            version=os.environ.get("DOCKER_API_VERSION", "auto"),
                            max_pool_size=pool_size,
                            num_pools=max(25, pool_size // 4),
                        )
                        print(f"Seeded R2E-Gym shared Docker client with max_pool_size={pool_size}")
        except Exception as e:
            print(f"WARN: could not pre-seed R2E-Gym Docker client with large pool: {e}")

    def _check_retrieval_connectivity(self):
        """Pre-flight check: verify retrieval server is reachable before launching trajectories."""
        retrieval_url = os.environ.get("RETRIEVAL_SERVER_URL", "")
        if not retrieval_url:
            return  # No retrieval server configured, skip check

        try:
            import httpx
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{retrieval_url.rstrip('/')}/health")
                response.raise_for_status()
            print(f"Retrieval health check passed (host={retrieval_url})")
        except Exception as e:
            raise RuntimeError(
                f"Retrieval server is unreachable at {retrieval_url}: {e}\n"
                f"Please ensure the retrieval server is running and accessible. "
                f"You can verify with: curl {retrieval_url.rstrip('/')}/health"
            ) from e

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        assert self.agent_class is not None and self.env_class is not None, "Agent and environment classes must be provided"

        # Pre-flight Docker health check: fail fast before creating any envs
        self._check_docker_connectivity()

        # Pre-flight retrieval server health check: fail fast before creating any envs
        self._check_retrieval_connectivity()

        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments and agents concurrently in a single executor pass
        n_items = len(env_args)
        envs = [None] * n_items
        agents = [None] * n_items
        with ThreadPoolExecutor(max_workers=min(n_items * 2, 256)) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(n_items)]
            agent_futures = [executor.submit(_create_agent, i) for i in range(n_items)]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from rllm.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            print(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                # Extract data_source from extra_info if not already a top-level field
                if "data_source" not in batch.non_tensor_batch and "extra_info" in batch.non_tensor_batch:
                    extra_infos = batch.non_tensor_batch["extra_info"]
                    data_sources = np.array(
                        [ei.get("data_source", "unknown") if isinstance(ei, dict) else "unknown" for ei in extra_infos],
                        dtype=object,
                    )
                    batch.non_tensor_batch["data_source"] = data_sources
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"], data_sources=batch.non_tensor_batch.get("data_source"))

                        if "idxs" in final_gen_batch_output.non_tensor_batch:
                            valid_indices = np.unique(final_gen_batch_output.non_tensor_batch["idxs"])
                            if len(valid_indices) == 0:
                                print("No valid trajectories found in this batch (Stepwise). Skipping...")
                                continue
                            if len(valid_indices) < len(batch.batch):
                                batch = batch.select_idxs(valid_indices)

                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info, batch=batch)

                        if "idxs" in final_gen_batch_output.non_tensor_batch:
                            valid_indices = final_gen_batch_output.non_tensor_batch["idxs"]
                            if len(valid_indices) == 0:
                                print("No valid trajectories found in this batch. Skipping...")
                                continue
                            if len(valid_indices) < len(batch.batch):
                                batch = batch.select_idxs(valid_indices)

                        batch = batch.union(final_gen_batch_output)
                        # Drop the internal per-trajectory raw metric lists before merging
                        # into the global training metrics (they're consumed by eval only).
                        generate_metrics.pop("_raw_traj_metrics", None)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        solve_no_variance = 0
                        reward_std_values = []
                        reward_mean_values = []
                        reward_range_values = []
                        verifier_missing_groups = 0
                        high_variance_groups = 0
                        verifier_missing_mask = None
                        if "reward_metadata" in batch.non_tensor_batch:
                            verifier_missing_mask = np.array([
                                0 if isinstance(metadata, dict) and metadata else 1
                                for metadata in batch.non_tensor_batch["reward_metadata"]
                            ], dtype=np.int32)
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            uid_reward_mean = uid_rewards.mean().detach().item()
                            uid_reward_std = uid_rewards.std(unbiased=False).detach().item()
                            uid_reward_min = uid_rewards.min().detach().item()
                            uid_reward_max = uid_rewards.max().detach().item()
                            reward_mean_values.append(uid_reward_mean)
                            reward_std_values.append(uid_reward_std)
                            reward_range_values.append(uid_reward_max - uid_reward_min)

                            group_missing_verifier = False
                            if verifier_missing_mask is not None:
                                group_missing_verifier = bool(verifier_missing_mask[uid_mask].any())
                                if group_missing_verifier:
                                    verifier_missing_groups += 1

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif uid_reward_std < 1e-6:
                                # All samples have same partial reward — GRPO advantage is 0
                                solve_no_variance += 1
                                if self.config.rllm.rejection_sample.get("filter_zero_variance", False):
                                    valid_mask[uid_mask] = False

                            if self.config.rllm.rejection_sample.get("filter_verifier_missing", False) and group_missing_verifier:
                                valid_mask[uid_mask] = False

                            max_uid_reward_std = self.config.rllm.rejection_sample.get("max_uid_reward_std", None)
                            if self.config.rllm.rejection_sample.get("filter_high_variance", False) and max_uid_reward_std is not None and uid_reward_std > max_uid_reward_std:
                                valid_mask[uid_mask] = False
                                high_variance_groups += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_no_variance"] = solve_no_variance
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all - solve_no_variance
                        if reward_mean_values:
                            metrics["batch/uid_reward_mean_mean"] = float(np.mean(reward_mean_values))
                            metrics["batch/uid_reward_mean_min"] = float(np.min(reward_mean_values))
                            metrics["batch/uid_reward_mean_max"] = float(np.max(reward_mean_values))
                        if reward_std_values:
                            metrics["batch/uid_reward_std_mean"] = float(np.mean(reward_std_values))
                            metrics["batch/uid_reward_std_max"] = float(np.max(reward_std_values))
                        if reward_range_values:
                            metrics["batch/uid_reward_range_mean"] = float(np.mean(reward_range_values))
                            metrics["batch/uid_reward_range_max"] = float(np.max(reward_range_values))
                        metrics["batch/verifier_missing_groups"] = verifier_missing_groups
                        metrics["batch/high_variance_groups"] = high_variance_groups

                        # Per-data-source reward metrics
                        batch_data_sources = batch.non_tensor_batch.get("data_source")
                        if batch_data_sources is not None:
                            from collections import defaultdict
                            source_rewards = defaultdict(list)
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_indices = np.where(uid_mask)[0]
                                src = batch_data_sources[uid_indices[0]]
                                uid_reward = reward_tensor[uid_mask].sum(-1).mean().detach().item()
                                source_rewards[src].append(uid_reward)
                            for src, rewards in source_rewards.items():
                                metrics[f"critic/rewards/{src}"] = float(np.mean(rewards))

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.rllm.stepwise_advantage.enable:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # For GDPO, also set up the component rewards
                        if self.config.algorithm.adv_estimator == "gdpo":
                            if "token_level_scores_base" in batch.batch:
                                batch.batch["token_level_rewards_base"] = batch.batch["token_level_scores_base"]
                            if "token_level_scores_bonus" in batch.batch:
                                batch.batch["token_level_rewards_bonus"] = batch.batch["token_level_scores_bonus"]

                        if self.config.rllm.stepwise_advantage.enable:
                            if self.config.rllm.stepwise_advantage.mode == "per_step":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        # Accumulate per-trajectory metric lists across all validation batches so we
        # can surface aggregates like val/steps/{mcp,search,cli}_{mean,min,max}.
        eval_traj_metrics: dict[str, list] = {}

        # Get max_val_num from config (-1 means use all batches)
        max_val_num = self.config.actor_rollout_ref.rollout.val_kwargs.get("max_val_num", -1)

        for batch_idx, test_data in enumerate(self.val_dataloader):
            # Break if we've reached the maximum number of validation batches
            if max_val_num > 0 and batch_idx >= max_val_num:
                break

            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            # Extract data_source from extra_info if not already a top-level field
            if "data_source" not in test_batch.non_tensor_batch and "extra_info" in test_batch.non_tensor_batch:
                extra_infos = test_batch.non_tensor_batch["extra_info"]
                data_sources = np.array(
                    [ei.get("data_source", "unknown") if isinstance(ei, dict) else "unknown" for ei in extra_infos],
                    dtype=object,
                )
                test_batch.non_tensor_batch["data_source"] = data_sources
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"], data_sources=test_batch.non_tensor_batch.get("data_source"))
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, gen_metrics = self.generate_agent_trajectory(meta_info=test_batch.meta_info, batch=test_batch, is_eval=True)
                # Collect raw per-trajectory metric lists (e.g. steps/mcp, steps/search,
                # steps/cli) so we can aggregate across all validation batches below.
                if isinstance(gen_metrics, dict):
                    raw = gen_metrics.get("_raw_traj_metrics")
                    if isinstance(raw, dict):
                        for k, vs in raw.items():
                            eval_traj_metrics.setdefault(k, []).extend(vs)

            # Filter test_batch to only valid indices (some trajectories may have been dropped)
            if "idxs" in test_output_gen_batch.non_tensor_batch:
                valid_indices = test_output_gen_batch.non_tensor_batch["idxs"]
                if len(valid_indices) == 0:
                    print("No valid trajectories found in this validation batch. Skipping...")
                    continue
                if len(valid_indices) < len(test_batch.batch):
                    test_batch = test_batch.select_idxs(valid_indices)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/{data_source}/pass@1"] = np.mean(rewards_array)

        if n_val_samples > 1:
            for data_source, pass_rates in data_source_uid_pass_rates.items():
                pass_k_lst = []
                for uid, pass_score in pass_rates.items():
                    pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
                metric_dict[f"val/{data_source}/pass@{n_val_samples}"] = np.mean(pass_k_lst)

        # Per-task-type step aggregates across all eval batches
        # (val/steps/{mcp,search,cli}_{mean,min,max} and overall val/steps_*).
        for k in ("steps", "steps/mcp", "steps/search", "steps/cli"):
            vs = eval_traj_metrics.get(k)
            if not vs:
                continue
            arr = np.array(vs, dtype=float)
            metric_dict[f"val/{k}_mean"] = float(arr.mean())
            metric_dict[f"val/{k}_min"] = float(arr.min())
            metric_dict[f"val/{k}_max"] = float(arr.max())

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None, batch=None, is_eval=False):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}

        dropped_trajectories = []
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    if trajectory.get("dropped", False):
                        dropped_trajectories.append(trajectory)
                    else:
                        trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")

        # Dump dropped trajectories
        dropped_dump = []
        if dropped_trajectories:
            for traj in dropped_trajectories:
                idx = traj["idx"]
                # Ensure we can access uid
                if batch is not None and "uid" in batch.non_tensor_batch:
                    u_id = batch.non_tensor_batch["uid"][idx]
                else:
                    u_id = "unknown"

                messages = traj.get("chat_completions", [])

                # Find the prompt
                prompt = ""
                for msg in messages:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                        break

                dropped_dump.append(
                    {
                        "uuid": str(u_id),
                        "prompt": prompt,
                        "data_source": batch.non_tensor_batch.get("data_source", ["unknown"] * (idx + 1))[idx] if batch is not None else "unknown",
                        "steps": len([turn for turn in messages if turn["role"] not in ["system", "user"]]),
                        "reward": None,
                        "termination_reason": traj.get("termination_reason"),
                        "trajectory": messages,
                        "debug": {},
                    }
                )

            subdir = "evals_trajectory" if is_eval else "train_trajectory"
            save_dir = os.path.join(self.config.trainer.default_local_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            # Dropped trajectories will be merged in _transform_agent_trajectories

        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        # Guard: if all trajectories failed/were skipped, return empty result
        # so fit_agent can detect empty idxs and `continue` to the next iteration.
        if len(trajectories) == 0:
            print(f"All {len(dropped_trajectories)} trajectories failed. Returning empty batch.")
            if dropped_dump:
                subdir = "evals_trajectory" if is_eval else "train_trajectory"
                save_dir = os.path.join(self.config.trainer.default_local_dir, subdir)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"global_steps_{self.global_steps}.json")
                merged_data = {
                    "traj_stats": {},
                    "accept_traj": [],
                    "reject_traj": dropped_dump,
                }
                with open(file_path, "w") as f:
                    json.dump(merged_data, f, ensure_ascii=False, indent=4, cls=_SafeEncoder)
            empty_output = DataProto.from_dict(tensors={}, non_tensors={"idxs": np.array([])})
            metrics = {"traj/accept_rate": 0.0}
            return empty_output, metrics

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories, dropped_dump=dropped_dump, batch=batch, is_eval=is_eval)

        total_trajectories = len(trajectories) + len(dropped_trajectories)
        metrics["traj/accept_rate"] = len(trajectories) / total_trajectories if total_trajectories > 0 else 0.0

        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None, data_sources=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []

        dropped_trajectories = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                if trajectory.get("dropped", False):
                    dropped_trajectories.append(trajectory)
                else:
                    steps.append(trajectory)

        # Dump dropped trajectories (Stepwise)
        dropped_dump = []
        if dropped_trajectories:
            for traj in dropped_trajectories:
                idx = traj["idx"]
                if uids is not None and len(uids) > idx:
                    u_id = uids[idx]
                else:
                    u_id = "unknown"

                messages = traj.get("chat_completions", [])

                # Find the prompt
                prompt = ""
                for msg in messages:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                        break

                dropped_dump.append({"uuid": str(u_id), "prompt": prompt, "data_source": data_sources[idx] if data_sources is not None and len(data_sources) > idx else "unknown", "steps": len([turn for turn in messages if turn["role"] not in ["system", "user"]]), "reward": None, "termination_reason": traj.get("termination_reason"), "trajectory": messages, "debug": {}})

            save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
            os.makedirs(save_dir, exist_ok=True)
            # Dropped trajectories will be merged in _transform_agent_steps

        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids, dropped_dump=dropped_dump, data_sources=data_sources)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict], dropped_dump: list[dict] = None, batch: DataProto = None, is_eval: bool = False):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.
            dropped_dump (list of dict): List of dropped trajectories.
            batch (DataProto): The original batch of data.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length, masked_whiten

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        traj_base_rewards = []  # Store base rewards for GDPO
        traj_bonus_rewards = []  # Store bonus rewards (tool_call + step_bonus) for GDPO
        chat_completions = []
        traj_metrics = []
        reward_metadata_list = []
        metrics = {}
        valid_indices = []

        for idx, traj in enumerate(trajectories):
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_score = traj["trajectory_reward"]
            # Env-error-driven action loops set reward=None to signal "mask from loss".
            # Zero out the response mask so this trajectory contributes no gradient.
            if traj_score is None:
                all_masks_list[-1] = torch.zeros_like(traj["response_masks"])
                traj_score = 0.0
            traj_scores.append(traj_score)

            # Extract reward components from metadata for GDPO
            reward_metadata = traj.get("reward_metadata", {})
            base_reward = reward_metadata.get("base_reward", traj["trajectory_reward"])
            tool_call_reward = reward_metadata.get("tool_call_reward", 0.0)
            step_bonus = reward_metadata.get("step_bonus", 0.0)

            # Store base reward and bonus rewards separately
            traj_base_rewards.append(base_reward)
            traj_bonus_rewards.append(tool_call_reward + step_bonus)
            reward_metadata_list.append(reward_metadata)

            trajectories_w_metadata = traj["chat_completions"].copy()

            original_idx = traj["idx"]
            valid_indices.append(original_idx)

            trajectories_w_metadata.append(
                {
                    "steps": len([turn for turn in trajectories_w_metadata if turn["role"] not in ["system", "user"]]),
                    "reward": traj["trajectory_reward"].item() if hasattr(traj["trajectory_reward"], "item") else float(traj["trajectory_reward"]),
                    "ground_truth": batch.non_tensor_batch.get("extra_info")[original_idx].get("ground_truth", ""),
                }
            )
            chat_completions.append(trajectories_w_metadata)
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        # Collect all unique keys from all trajectories to handle missing metrics
        all_keys = set()
        for d in traj_metrics:
            all_keys.update(d.keys())

        # Use .get() to handle missing keys gracefully
        traj_metrics = {k: [d.get(k, None) for d in traj_metrics] for k in all_keys}

        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if isinstance(v, (int, float)) and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Surface the raw per-trajectory metric lists so callers (e.g. _validate_agent)
        # can aggregate them across multiple batches and emit val/ metrics.
        metrics["_raw_traj_metrics"] = {
            k: [v for v in vs if isinstance(v, (int, float)) and v >= 0]
            for k, vs in traj_metrics.items()
        }

        verifier_pass_rates = [m.get("pass_rate") for m in reward_metadata_list if isinstance(m, dict) and m.get("pass_rate") is not None]
        verifier_resolved = [1.0 if m.get("resolved") else 0.0 for m in reward_metadata_list if isinstance(m, dict) and "resolved" in m]
        verifier_errors = [1.0 if m.get("verifier_error") else 0.0 for m in reward_metadata_list if isinstance(m, dict)]
        verifier_missing = [1.0 if not (isinstance(m, dict) and m) else 0.0 for m in reward_metadata_list]
        tests_passed = [m.get("tests_passed") for m in reward_metadata_list if isinstance(m, dict) and m.get("tests_passed") is not None]
        tests_failed = [m.get("tests_failed") for m in reward_metadata_list if isinstance(m, dict) and m.get("tests_failed") is not None]
        tests_total = [m.get("tests_total") for m in reward_metadata_list if isinstance(m, dict) and m.get("tests_total") is not None]

        if verifier_pass_rates:
            metrics["traj/verifier_pass_rate_mean"] = float(np.mean(verifier_pass_rates))
            metrics["traj/verifier_pass_rate_max"] = float(np.max(verifier_pass_rates))
        if verifier_resolved:
            metrics["traj/verifier_resolved_rate"] = float(np.mean(verifier_resolved))
        if verifier_errors:
            metrics["traj/verifier_error_rate"] = float(np.mean(verifier_errors))
        if verifier_missing:
            metrics["traj/verifier_missing_rate"] = float(np.mean(verifier_missing))
        if tests_passed:
            metrics["traj/tests_passed_mean"] = float(np.mean(tests_passed))
        if tests_failed:
            metrics["traj/tests_failed_mean"] = float(np.mean(tests_failed))
        if tests_total:
            metrics["traj/tests_total_mean"] = float(np.mean(tests_total))

        # Save chat completions and stats to files
        subdir = "evals_trajectory" if is_eval else "train_trajectory"
        save_dir = os.path.join(self.config.trainer.default_local_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)

        # Dump trajectories with uuid and prompt
        traj_dump = []
        for traj in trajectories:
            idx = traj["idx"]
            u_id = batch.non_tensor_batch["uid"][idx]
            messages = traj["chat_completions"]

            # Find the prompt. Usually the first user message.
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt = msg["content"]
                    break

            traj_dump.append({
                "uuid": str(u_id),
                "prompt": prompt,
                "data_source": batch.non_tensor_batch.get("data_source", ["unknown"] * (idx + 1))[idx] if batch is not None else "unknown",
                "steps": len([turn for turn in messages if turn["role"] not in ["system", "user"]]),
                "reward": traj["trajectory_reward"].item() if hasattr(traj["trajectory_reward"], "item") else float(traj["trajectory_reward"]),
                "termination_reason": traj.get("termination_reason"),
                "trajectory": messages,
                "debug": {
                    "verification": traj.get("reward_debug", {}),
                    "reward_metadata": traj.get("reward_metadata", {}),
                    "ground_truth": batch.non_tensor_batch.get("extra_info")[idx].get("ground_truth", "") if batch is not None else "",
                    "metrics": traj.get("metrics", {}),
                    "exception": traj.get("exception", ""),
                },
            })

        # Collect termination reason statistics
        all_reasons = [
            "ENV_DONE",
            "TIMEOUT",
            "MAX_STEPS",
            "TRUNCATION",
            "PROMPT_TRUNCATION",
            "ABNORMAL_PARSE_ERROR",
            "ABNORMAL_TOOL_BURST",
            "ABNORMAL_REPEATED_QUERY",
            "INVALID_REACT_STRUCTURE",
            "INVALID_FINAL_STEP",
            "ENV_TIMEOUT",
            "UNKNOWN",
        ]

        success_stats = {r: 0 for r in all_reasons}
        failure_stats = {r: 0 for r in all_reasons}
        negative_reward_stats = {r: 0 for r in all_reasons}
        total_stats = {r: 0 for r in all_reasons}

        abnormal_reasons = {
            "ABNORMAL_PARSE_ERROR",
            "ABNORMAL_TOOL_BURST",
            "ABNORMAL_REPEATED_QUERY",
            "INVALID_REACT_STRUCTURE",
            "INVALID_FINAL_STEP",
            "ENV_TIMEOUT",
        }

        all_dumps = traj_dump + (dropped_dump or [])

        # Per-source termination reason and verifier_error counters
        from collections import defaultdict as _dd
        source_termination: dict = _dd(lambda: _dd(int))
        source_verifier_error: dict = _dd(list)
        source_env_loop: dict = _dd(int)

        for traj in all_dumps:
            reason = traj.get("termination_reason") or "UNKNOWN"
            reward = traj.get("reward")
            if reward is None:
                reward = 0
            src = traj.get("data_source", "unknown")

            total_stats[reason] = total_stats.get(reason, 0) + 1
            if reward >= 1.0:
                success_stats[reason] = success_stats.get(reason, 0) + 1
            elif reason in abnormal_reasons:
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
            else:
                negative_reward_stats[reason] = negative_reward_stats.get(reason, 0) + 1

            source_termination[src][reason] += 1
            dbg = traj.get("debug", {})
            verif = dbg.get("verification", {}) if isinstance(dbg, dict) else {}
            if isinstance(verif, dict) and verif:
                source_verifier_error[src].append(1.0 if verif.get("verifier_error") else 0.0)
            if reason == "ABNORMAL_ACTION_LOOP":
                source_env_loop[src] += 1

        # Save merged chat completions and stats
        file_path = os.path.join(save_dir, f"global_steps_{self.global_steps}.json")
        traj_stats_data = {
            # "success": success_stats,
            # "failure": failure_stats,
            # "negative_reward": negative_reward_stats,
            "total": total_stats,
        }
        merged_data = {
            "traj_stats": traj_stats_data["total"],
            "accept_traj": traj_dump,
            "reject_traj": dropped_dump or [],
        }
        with open(file_path, "w") as f:
            print(f"Saving merged trajectories and stats to {file_path}")
            json.dump(merged_data, f, ensure_ascii=False, indent=4, cls=_SafeEncoder)

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        base_reward_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        bonus_reward_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score
                base_reward_batch[i, resp_len - 1] = traj_base_rewards[i]
                bonus_reward_batch[i, resp_len - 1] = traj_bonus_rewards[i]

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "token_level_scores_base": base_reward_batch,  # Base reward for GDPO
            "token_level_scores_bonus": bonus_reward_batch,  # Bonus reward for GDPO
            "response_mask": traj_mask,
        }

        non_tensor_batch = {
            "idxs": np.array(valid_indices),
            "reward_metadata": np.array(reward_metadata_list, dtype=object),
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch))

        # Per-source termination and verifier_error metrics
        for src, reason_counts in source_termination.items():
            for reason, cnt in reason_counts.items():
                metrics[f"traj/{src}/termination/{reason}"] = cnt
        for src, errs in source_verifier_error.items():
            metrics[f"traj/{src}/verifier_error_rate"] = float(np.mean(errs)) if errs else 0.0
        for src, cnt in source_env_loop.items():
            metrics[f"traj/{src}/env_loop_terminations"] = cnt

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray, dropped_dump: list[dict] = None, data_sources=None):
        from verl.utils.torch_functional import pad_sequence_to_length, masked_whiten

        overlong_filter = self.config.rllm.agent.get("overlong_filter", False)
        overlong_reasons = {"TRUNCATION", "MAX_STEPS", "TIMEOUT"}

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        all_steps_masked_out = []  # whether this step should be masked out due to overlong filter
        training_rewards = []
        training_base_rewards = []  # Store base rewards for GDPO
        training_bonus_rewards = []  # Store bonus rewards for GDPO
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            # Extract reward components from metadata for GDPO
            reward_metadata = episode.get("reward_metadata", {})
            base_reward = reward_metadata.get("base_reward", training_reward)
            tool_call_reward = reward_metadata.get("tool_call_reward", 0.0)
            step_bonus = reward_metadata.get("step_bonus", 0.0)

            training_base_rewards.append(base_reward)
            training_bonus_rewards.append(tool_call_reward + step_bonus)
            termination_reason = episode.get("termination_reason") or "UNKNOWN"

            # Mask out overlong trajectories
            masked_out = overlong_filter and termination_reason in overlong_reasons

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])
            all_steps_masked_out.extend([masked_out for _ in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]
        # apply overlong filter by zeroing out masked trajectories
        if overlong_filter:
            overlong_mask = torch.tensor(all_steps_masked_out, dtype=torch.bool).unsqueeze(1)
            traj_mask = traj_mask * (~overlong_mask).long()

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        base_reward_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        bonus_reward_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    base_reward_batch[step_index, resp_len - 1] = training_base_rewards[i]
                    bonus_reward_batch[step_index, resp_len - 1] = training_bonus_rewards[i]
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "token_level_scores_base": base_reward_batch,  # Base reward for GDPO
            "token_level_scores_bonus": bonus_reward_batch,  # Bonus reward for GDPO
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        # Save chat completions and stats
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)

        # Dump trajectories with uuid and prompt
        traj_dump = []
        for episode in steps:
            idx = episode["idx"]
            u_id = uids[idx]
            episode_steps = episode["steps"]

            # In stepwise mode, episode["steps"] is a list of dicts with "prompt" and "response"
            main_prompt = episode_steps[0]["prompt"] if episode_steps else ""

            traj_dump.append({
                "uuid": str(u_id),
                "prompt": main_prompt,
                "data_source": data_sources[idx] if data_sources is not None and len(data_sources) > idx else "unknown",
                "steps": len([turn for turn in episode_steps if turn["prompt"] not in ["system", "user"]]),
                "reward": episode["trajectory_reward"].item() if hasattr(episode["trajectory_reward"], "item") else float(episode["trajectory_reward"]),
                "termination_reason": episode.get("termination_reason"),
                "trajectory": episode_steps,
                "debug": {
                    "verification": episode.get("reward_debug", {}),
                    "reward_metadata": episode.get("reward_metadata", {}),
                    "metrics": episode.get("metrics", {}),
                    "exception": episode.get("exception", ""),
                },
            })

        # Collect termination reason statistics
        all_reasons = [
            "ENV_DONE",
            "TIMEOUT",
            "MAX_STEPS",
            "TRUNCATION",
            "PROMPT_TRUNCATION",
            "ABNORMAL_PARSE_ERROR",
            "ABNORMAL_TOOL_BURST",
            "ABNORMAL_REPEATED_QUERY",
            "INVALID_REACT_STRUCTURE",
            "INVALID_FINAL_STEP",
            "ENV_TIMEOUT",
            "UNKNOWN",
        ]

        success_stats = {r: 0 for r in all_reasons}
        failure_stats = {r: 0 for r in all_reasons}
        negative_reward_stats = {r: 0 for r in all_reasons}
        total_stats = {r: 0 for r in all_reasons}

        abnormal_reasons = {
            "ABNORMAL_PARSE_ERROR",
            "ABNORMAL_TOOL_BURST",
            "ABNORMAL_REPEATED_QUERY",
            "INVALID_REACT_STRUCTURE",
            "INVALID_FINAL_STEP",
            "ENV_TIMEOUT",
        }

        all_dumps = traj_dump + (dropped_dump or [])

        # Per-source termination reason and verifier_error counters
        from collections import defaultdict as _dd
        source_termination: dict = _dd(lambda: _dd(int))
        source_verifier_error: dict = _dd(list)
        source_env_loop: dict = _dd(int)

        for traj in all_dumps:
            reason = traj.get("termination_reason") or "UNKNOWN"
            reward = traj.get("reward")
            if reward is None:
                reward = 0
            src = traj.get("data_source", "unknown")

            total_stats[reason] = total_stats.get(reason, 0) + 1
            if reward >= 1.0:
                success_stats[reason] = success_stats.get(reason, 0) + 1
            elif reason in abnormal_reasons:
                failure_stats[reason] = failure_stats.get(reason, 0) + 1
            else:
                negative_reward_stats[reason] = negative_reward_stats.get(reason, 0) + 1

            source_termination[src][reason] += 1
            dbg = traj.get("debug", {})
            verif = dbg.get("verification", {}) if isinstance(dbg, dict) else {}
            if isinstance(verif, dict) and verif:
                source_verifier_error[src].append(1.0 if verif.get("verifier_error") else 0.0)
            if reason == "ABNORMAL_ACTION_LOOP":
                source_env_loop[src] += 1

        file_path = os.path.join(save_dir, f"global_steps_{self.global_steps}.json")
        traj_stats_data = {
            # "success": success_stats,
            # "failure": failure_stats,
            # "negative_reward": negative_reward_stats,
            "total": total_stats,
        }
        merged_data = {
            "traj_stats": traj_stats_data["total"],
            "accept_traj": traj_dump,
            "reject_traj": dropped_dump or [],
        }
        with open(file_path, "w") as f:
            print(f"Saving merged chat completions and stats (Stepwise) to {file_path}")
            json.dump(merged_data, f, ensure_ascii=False, indent=4, cls=_SafeEncoder)

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Per-source termination and verifier_error metrics (stepwise path)
        for src, reason_counts in source_termination.items():
            for reason, cnt in reason_counts.items():
                metrics[f"traj/{src}/termination/{reason}"] = cnt
        for src, errs in source_verifier_error.items():
            metrics[f"traj/{src}/verifier_error_rate"] = float(np.mean(errs)) if errs else 0.0
        for src, cnt in source_env_loop.items():
            metrics[f"traj/{src}/env_loop_terminations"] = cnt

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
