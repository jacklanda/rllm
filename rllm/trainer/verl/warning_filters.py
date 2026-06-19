"""Narrow warning/log filters for noisy training compatibility messages."""

from __future__ import annotations

import logging
import warnings


class _MessagePrefixFilter(logging.Filter):
    def __init__(self, *prefixes: str) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(message.startswith(prefix) for prefix in self._prefixes)


_VLLM_RAW_PROMPT_FILTER = _MessagePrefixFilter(
    "Passing raw prompts to InputProcessor is deprecated "
    "and will be removed in v0.18. You should instead pass "
    "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
)

_TORCH_FX_IS_TRACING_FILTER = _MessagePrefixFilter(
    "is_fx_tracing will return true for both fx.symbolic_trace and "
    "torch.export. Please use "
    "is_fx_tracing_symbolic_tracing() for specifically fx.symbolic_trace "
    "or torch.compiler.is_compiling() for specifically torch.export/compile."
)


def apply_verl_vllm_noise_filters() -> None:
    """Suppress known harmless third-party startup warnings.

    The Qwen2 tokenizer message is emitted while Verl probes for a processor
    for text-only models. vLLM's raw-prompt message is a deprecation warning
    from the server internals. PyTorch's FX tracing warning is emitted by
    third-party compatibility checks in worker startup. These are noisy for
    current training runs and do not require a behavior change in rLLM.
    """

    warnings.filterwarnings(
        "ignore",
        message=r"Failed to create processor: Unsupported processor type: Qwen2Tokenizer\. This may affect multimodal processing",
        category=UserWarning,
        module=r"verl\.utils\.tokenizer",
    )

    logger = logging.getLogger("vllm.v1.engine.input_processor")
    if not any(existing is _VLLM_RAW_PROMPT_FILTER for existing in logger.filters):
        logger.addFilter(_VLLM_RAW_PROMPT_FILTER)

    logger = logging.getLogger("torch.fx._symbolic_trace")
    if not any(existing is _TORCH_FX_IS_TRACING_FILTER for existing in logger.filters):
        logger.addFilter(_TORCH_FX_IS_TRACING_FILTER)
