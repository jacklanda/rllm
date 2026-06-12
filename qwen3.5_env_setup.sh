#!/usr/bin/env bash
set -euo pipefail

# Use the system CUDA 12.9 toolchain for source builds.
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Build helpers needed by flash-attn with --no-build-isolation.
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging ninja

# Current validated CUDA-12-compatible stack. --no-deps avoids resolver backtracking.
CORE_PACKAGES=(
  torch==2.10.0
  torchvision==0.25.0
  torchaudio==2.10.0
  triton==3.6.0
  cuda-bindings==12.9.4
  cuda-python==12.9.4
  vllm==0.19.1
  verl==0.7.0
  transformers==5.7.0
  numpy==1.26.4
  huggingface-hub==1.18.0
  tokenizers==0.22.2
  flashinfer-python==0.6.6
  flashinfer-cubin==0.6.6
  compressed-tensors==0.15.0.1
  depyf==0.20.0
  outlines-core==0.2.11
  xgrammar==0.2.1
  lm-format-enforcer==0.11.3
  llguidance==1.3.0
  numba==0.61.2
  llvmlite==0.44.0
)

"${PYTHON_BIN}" -m pip install --no-cache-dir --force-reinstall --no-deps "${CORE_PACKAGES[@]}"

# Patch verl for the Transformers 5.x rename.
VERL_PACKAGE_DIR="$("${PYTHON_BIN}" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("verl")
if spec is None or spec.origin is None:
    raise SystemExit("verl is not installed")

print(Path(spec.origin).parent)
PY
)"
find "${VERL_PACKAGE_DIR}" -type f -name "*.py" -exec perl -0pi -e 's/AutoModelForVision2Seq/AutoModelForImageTextToText/g' {} +

# Torch 2.10.0 CUDA runtime dependencies.
CUDA_RUNTIME_PACKAGES=(
  nvidia-cublas-cu12==12.8.4.1
  nvidia-cuda-cupti-cu12==12.8.90
  nvidia-cuda-nvrtc-cu12==12.8.93
  nvidia-cuda-runtime-cu12==12.8.90
  nvidia-cudnn-cu12==9.10.2.21
  nvidia-cufft-cu12==11.3.3.83
  nvidia-cufile-cu12==1.13.1.3
  nvidia-curand-cu12==10.3.9.90
  nvidia-cusolver-cu12==11.7.3.90
  nvidia-cusparse-cu12==12.5.8.93
  nvidia-cusparselt-cu12==0.7.1
  nvidia-nccl-cu12==2.27.5
  nvidia-nvjitlink-cu12==12.8.93
  nvidia-nvshmem-cu12==3.4.5
  nvidia-nvtx-cu12==12.8.90
)

"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade --force-reinstall "${CUDA_RUNTIME_PACKAGES[@]}"

# vLLM 0.19.1 does not require the old xformers wheel.
"${PYTHON_BIN}" -m pip uninstall -y xformers || true

# Build flash-attn against the active Torch and CUDA toolchain.
"${PYTHON_BIN}" -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3

# Final import check.
"${PYTHON_BIN}" -c "from flash_attn import *; from torch import *; from transformers import *; from vllm import *; from verl import *"
