from types import SimpleNamespace

from rllm.experimental.verl import utils


def test_maybe_hf_processor_skips_text_only_qwen_config(monkeypatch):
    calls = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"])

    def hf_processor(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("transformers.AutoConfig", AutoConfig)
    monkeypatch.setattr("verl.utils.hf_processor", hf_processor)

    assert utils.maybe_hf_processor("/tmp/model", trust_remote_code=True, backend="torchvision") is None
    assert calls == []


def test_maybe_hf_processor_loads_vl_config(monkeypatch):
    processor = object()

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(model_type="qwen2_5_vl", architectures=["Qwen2_5_VLForConditionalGeneration"])

    def hf_processor(*args, **kwargs):
        return processor

    monkeypatch.setattr("transformers.AutoConfig", AutoConfig)
    monkeypatch.setattr("verl.utils.hf_processor", hf_processor)

    assert utils.maybe_hf_processor("/tmp/model", trust_remote_code=True, backend="torchvision") is processor
