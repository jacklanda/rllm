#!/usr/bin/env python3
"""Convert a verl FSDP training checkpoint into a self-contained HuggingFace model.

A verl FSDP checkpoint directory (e.g. ``.../global_step_89/``) holds the model
sharded across ranks::

    fsdp_config.json                     # {"FSDP_version": 2, "world_size": 8}
    model_world_size_8_rank_{0..7}.pt    # DTensor shards of the model weights
    extra_state_world_size_8_rank_*.pt   # optimizer / rng / lr-scheduler state
    huggingface/                         # config.json, tokenizer.*, chat_template, ...

This script reconstructs the full (unsharded) weights from those rank files and
writes a standard HuggingFace directory -- ``config.json``, ``*.safetensors``,
``generation_config.json`` and every tokenizer artifact -- so the result can be
loaded directly with ``AutoModelForCausalLM.from_pretrained(<target_dir>)``.

It is a thin, defaulted wrapper around ``verl.model_merger.FSDPModelMerger`` so
the merge logic stays in sync with the verl version that produced the ckpt.

Examples
--------
Convert the bundled SFT checkpoint into ``global_step_89/hf_model`` (default)::

    python scripts/convert_to_hf.py

Convert an arbitrary checkpoint to a chosen directory and re-load it to verify::

    python scripts/convert_to_hf.py \
        --local_dir /path/to/global_step_N \
        --target_dir /path/to/output_hf \
        --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo-relative default so `python scripts/convert_to_hf.py` "just works".
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CKPT = _REPO_ROOT / "examples/sft_agent/checkpoints/qwen3_4b_thinking_2507_agent_sft/global_step_89"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--local_dir",
        type=str,
        default=str(_DEFAULT_CKPT),
        help="verl FSDP checkpoint dir holding model_world_size_*_rank_*.pt, " "fsdp_config.json and a huggingface/ subdir (default: the bundled " "qwen3_4b_thinking_2507_agent_sft global_step_89).",
    )
    p.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Output dir for the merged HF model. Default: <local_dir>/hf_model.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model config/tokenizer.",
    )
    p.add_argument(
        "--hf_upload_path",
        type=str,
        default=None,
        help="Optional HF Hub repo id to upload the merged model to (e.g. user/model).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="If uploading, create/use a private HF Hub repo.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After merging, reload the result with AutoModelForCausalLM + " "AutoTokenizer as a sanity check.",
    )
    return p.parse_args()


def _validate_checkpoint(local_dir: Path) -> None:
    """Fail early with a clear message if the dir is not an FSDP checkpoint."""
    if not local_dir.is_dir():
        sys.exit(f"[convert_to_hf] local_dir does not exist: {local_dir}")

    fsdp_cfg = local_dir / "fsdp_config.json"
    if not fsdp_cfg.exists():
        sys.exit(f"[convert_to_hf] missing {fsdp_cfg.name} in {local_dir}; this does " "not look like a verl FSDP checkpoint.")

    hf_cfg = local_dir / "huggingface"
    if not (hf_cfg / "config.json").exists():
        sys.exit(f"[convert_to_hf] missing {hf_cfg}/config.json; the FSDP merger reads " "the base model config/tokenizer from this subdir.")

    shards = sorted(local_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        sys.exit(f"[convert_to_hf] found no model_world_size_*_rank_*.pt shards in {local_dir}.")


def _patch_transformers_compat() -> None:
    """Shim symbols verl's merger imports but newer transformers has renamed.

    verl 0.6.1's ``base_model_merger`` imports ``AutoModelForVision2Seq`` at
    module load time. transformers >= 5.x renamed it to
    ``AutoModelForImageTextToText`` and dropped the old name, so the import
    crashes before any merge logic runs. We alias the old name to the new class
    (or a harmless stand-in) so the import succeeds; this model is a plain
    CausalLM, so the vision class is never actually used.
    """
    import transformers

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return
    alias = getattr(
        transformers,
        "AutoModelForImageTextToText",
        getattr(transformers, "AutoModelForCausalLM", None),
    )
    if alias is not None:
        transformers.AutoModelForVision2Seq = alias


def _verify(target_dir: Path, trust_remote_code: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[convert_to_hf] verifying merged model in {target_dir} ...")
    tok = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        target_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    n_params = sum(p.numel() for p in model.parameters())
    has_template = bool(getattr(tok, "chat_template", None))
    print(f"[convert_to_hf] OK: loaded {type(model).__name__} " f"({n_params / 1e9:.2f}B params), vocab={len(tok)}, " f"chat_template={'present' if has_template else 'MISSING'}")
    del model


def main() -> None:
    args = parse_args()

    local_dir = Path(args.local_dir).resolve()
    target_dir = Path(args.target_dir).resolve() if args.target_dir else local_dir / "hf_model"

    _validate_checkpoint(local_dir)

    # Imported here so --help works even outside the training env.
    _patch_transformers_compat()
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    target_dir.mkdir(parents=True, exist_ok=True)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(local_dir),
        # The merger reads the base config/tokenizer from <local_dir>/huggingface.
        hf_model_config_path=str(local_dir / "huggingface"),
        target_dir=str(target_dir),
        trust_remote_code=args.trust_remote_code,
        hf_upload_path=args.hf_upload_path,
        private=args.private,
    )

    print(f"[convert_to_hf] merging FSDP shards from {local_dir}")
    print(f"[convert_to_hf] writing HuggingFace model to {target_dir}")

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

    print(f"[convert_to_hf] done. Load it with:")
    print(f"    AutoModelForCausalLM.from_pretrained('{target_dir}')")

    if args.verify:
        _verify(target_dir, args.trust_remote_code)


if __name__ == "__main__":
    main()
