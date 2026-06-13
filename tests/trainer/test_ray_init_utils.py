import importlib.util
import warnings
from pathlib import Path


def _load_ray_init_utils():
    module_path = Path(__file__).resolve().parents[2] / "rllm" / "trainer" / "ray_init_utils.py"
    spec = importlib.util.spec_from_file_location("rllm_ray_init_utils_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _load_ray_sitecustomize():
    module_path = Path(__file__).resolve().parents[2] / "rllm" / "trainer" / "_ray_compat" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("rllm_ray_sitecustomize_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_get_ray_init_settings_defaults_to_local_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert "address" not in settings
    assert settings["include_dashboard"] is False


def test_get_ray_init_settings_attaches_when_ray_address_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"
    assert settings["include_dashboard"] is False


def test_get_ray_init_settings_attaches_when_ray_current_cluster_file_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    marker = tmp_path / "ray_current_cluster"
    marker.write_text("dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: marker)

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"
    assert settings["include_dashboard"] is False


def test_get_ray_init_settings_ignores_other_users_ray_current_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    ray_init_utils = _load_ray_init_utils()

    class _Stat:
        st_uid = 999999

    class _FakePath:
        def exists(self):
            return True

        def stat(self):
            return _Stat()

    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: _FakePath())
    monkeypatch.setattr(ray_init_utils.os, "getuid", lambda: 123456)

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "local"
    assert settings["include_dashboard"] is False


def test_config_address_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    class Cfg:
        ray_init = {"address": "ray://explicit"}

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(Cfg())
    assert settings["address"] == "ray://explicit"


def test_config_include_dashboard_overrides_default(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    class Cfg:
        ray_init = {"include_dashboard": True}

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(Cfg())
    assert "address" not in settings
    assert settings["include_dashboard"] is True


def test_get_ray_init_settings_adds_ray_subprocess_compat_path(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/existing")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    ray_init_utils.get_ray_init_settings(config=None)
    ray_init_utils.get_ray_init_settings(config=None)

    compat_path = str(ray_init_utils._ray_compat_sitecustomize_dir())
    entries = ray_init_utils.os.environ["PYTHONPATH"].split(ray_init_utils.os.pathsep)
    assert entries[0] == compat_path
    assert entries.count(compat_path) == 1
    assert "/existing" in entries


def test_ray_subprocess_compat_silences_transformers_use_return_dict_warning(caplog):
    try:
        from transformers import PretrainedConfig
    except Exception:
        return

    _load_ray_sitecustomize()
    config = PretrainedConfig(return_dict=False)

    caplog.set_level("WARNING", logger="transformers")
    caplog.clear()

    assert config.use_return_dict is False
    assert not [record for record in caplog.records if "use_return_dict" in record.getMessage()]


def test_ray_subprocess_compat_suppresses_torch_inductor_online_softmax_warning():
    _load_ray_sitecustomize()
    message = (
        "Online softmax is disabled on the fly since Inductor decides to\n"
        "split the reduction. Cut an issue to PyTorch if this is an\n"
        "important use case and you want to speed it up with online\n"
        "softmax."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn_explicit(
            message,
            UserWarning,
            "/tmp/torch/_inductor/lowering.py",
            7627,
            module="torch._inductor.lowering",
        )

    assert caught == []
