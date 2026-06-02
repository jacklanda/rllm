from rllm.trainer.sft_metrics import normalize_sft_lr_metrics


def test_normalize_sft_lr_metrics_converts_verl_scaled_lr():
    metrics = {
        "train/loss": 0.5,
        "train/lr(1e-3)": 0.0005882352941176471,
    }

    normalized = normalize_sft_lr_metrics(metrics)

    assert "train/lr(1e-3)" not in normalized
    assert normalized["train/lr"] == 5.882352941176471e-7
    assert normalized["train/loss"] == 0.5


def test_normalize_sft_lr_metrics_keeps_existing_actual_lr():
    metrics = {
        "train/lr": 1e-5,
        "train/lr(1e-3)": 0.01,
    }

    normalized = normalize_sft_lr_metrics(metrics)

    assert normalized["train/lr"] == 1e-5
    assert "train/lr(1e-3)" not in normalized
