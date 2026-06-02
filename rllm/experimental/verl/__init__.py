from .dataclass import AccumulatedData, ProcessedStepData
from .transform import transform_episodes_to_dataproto, transform_trajectory_groups_to_dataproto, update_dataproto_with_advantages

__all__ = [
    # data transformation
    "transform_episodes_to_dataproto",
    "transform_trajectory_groups_to_dataproto",
    "update_dataproto_with_advantages",
    # backend
    "VerlBackend",
    # dataclass
    "AccumulatedData",
    "ProcessedStepData",
]


def __getattr__(name):
    if name == "VerlBackend":
        from .verl_backend import VerlBackend

        return VerlBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
