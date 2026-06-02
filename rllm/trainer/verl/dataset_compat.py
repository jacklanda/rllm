def patch_rlhf_dataset_answer_norm():
    import datasets
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if getattr(RLHFDataset, "_rllm_answer_norm_patched", False):
        return

    def patched(self):
        dataframes = []
        for parquet_file in self.data_files:
            # Use Dataset.from_parquet to avoid HF datasets script-resolution HEAD requests to S3.
            df = datasets.Dataset.from_parquet(parquet_file)
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

        common_cols = set(dataframes[0].column_names)
        for df in dataframes[1:]:
            common_cols &= set(df.column_names)
        dataframes = [df.select_columns(list(common_cols)) for df in dataframes]
        self.dataframe = datasets.concatenate_datasets(dataframes)

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
    RLHFDataset._rllm_answer_norm_patched = True
