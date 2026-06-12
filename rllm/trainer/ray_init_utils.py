"""Utilities for initializing Ray consistently.

Issue #166 reports that in some environments (notably Docker), a child process may
call `ray.init(namespace=...)` and accidentally start a fresh local Ray cluster
instead of attaching to the already-running one. This can lead to confusing
failures where named actors appear to be missing.

This module centralizes the logic for selecting Ray init parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _ray_current_cluster_path() -> Path:
    # Default location Ray uses to store the current cluster address.
    # See Ray docs and common troubleshooting guides.
    return Path("/tmp/ray/ray_current_cluster")


def _ray_compat_sitecustomize_dir() -> Path:
    return Path(__file__).resolve().parent / "_ray_compat"


def _prepend_pythonpath(path: Path) -> None:
    path_str = str(path)
    pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if path_str not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *entries])


def enable_ray_subprocess_compat_shims() -> None:
    """Expose small compatibility patches to Python subprocesses Ray starts."""

    _prepend_pythonpath(_ray_compat_sitecustomize_dir())


def get_default_ray_address() -> str | None:
    """Choose the safest implicit Ray address for the current machine."""

    if os.getenv("RAY_ADDRESS"):
        return "auto"

    try:
        cluster_path = _ray_current_cluster_path()
        if not cluster_path.exists():
            return None

        # On shared machines, `/tmp/ray/ray_current_cluster` is often left
        # behind by another user. Ray itself may auto-attach to that marker
        # even if we omit `address`, so explicitly force a local cluster when
        # the marker does not belong to the current user.
        if cluster_path.stat().st_uid != os.getuid():
            return "local"

        return "auto"
    except Exception:
        return None


def get_ray_init_settings(config: Any | None = None) -> dict[str, Any]:
    """Build kwargs for `ray.init(...)` from config + environment.

    Notes:
    - If `config.ray_init.address` is set, we pass it through verbatim.
    - Ray's dashboard is optional for training and can fail on unrelated
      observability dependency mismatches, so it is disabled by default. Set
      `ray_init.include_dashboard=true` to opt in.
    - Otherwise, if we detect a running cluster owned by the current user (or
      `RAY_ADDRESS` is set), we use `address="auto"` to attach.
    - If we only detect another user's stale cluster marker, we explicitly set
      `address="local"` to avoid Ray's own implicit auto-attach heuristic.
    - If none of the above applies, we return no `address`, so Ray will start a
      local cluster.
    """

    enable_ray_subprocess_compat_shims()

    settings: dict[str, Any] = {}

    if config is not None and hasattr(config, "ray_init"):
        for k, v in config.ray_init.items():
            if v is not None:
                settings[k] = v

    settings.setdefault("include_dashboard", False)

    # Prefer explicit address from config.
    if "address" in settings:
        return settings

    default_address = get_default_ray_address()
    if default_address is not None:
        settings["address"] = default_address

    return settings
