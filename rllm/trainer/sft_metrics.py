def normalize_sft_lr_metrics(metrics: dict) -> dict:
    """Convert VERL's scaled SFT LR metric into the actual learning rate."""
    scaled_lr_key = "train/lr(1e-3)"
    actual_lr_key = "train/lr"

    if scaled_lr_key in metrics:
        scaled_lr = metrics.pop(scaled_lr_key)
        if actual_lr_key not in metrics:
            metrics[actual_lr_key] = scaled_lr * 1e-3
    return metrics
