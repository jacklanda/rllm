import rllm.experimental.verl.patch as verl_patch


def test_verl_ray_worker_patch_sets_node_local_rank_and_deferred_cuda_binding_env(monkeypatch):
    from verl.single_controller.ray import RayWorkerGroup

    captured = {}

    def fake_create_worker(
        self,
        rank,
        pg_idx,
        pg,
        local_rank,
        resource_pool,
        ray_cls_with_init,
        worker_env,
        detached,
    ):
        captured["worker_env"] = worker_env

    monkeypatch.setattr(RayWorkerGroup, "_create_worker", fake_create_worker)
    monkeypatch.setenv("RLLM_CUDA_VISIBLE_DEVICES", "3,4,5,6")
    monkeypatch.setattr(verl_patch, "_VERL_RAY_LOCAL_RANK_ENV_PATCHED", False)
    verl_patch.patch_verl_ray_worker_local_rank_env()

    class ResourcePool:
        store = [8]

    RayWorkerGroup._create_worker(
        object(),
        rank=3,
        pg_idx=0,
        pg=object(),
        local_rank=3,
        resource_pool=ResourcePool(),
        ray_cls_with_init=object(),
        worker_env={"EXISTING": "1"},
        detached=False,
    )

    assert captured["worker_env"]["EXISTING"] == "1"
    assert captured["worker_env"]["LOCAL_RANK"] == "3"
    assert captured["worker_env"]["LOCAL_WORLD_SIZE"] == "8"
    assert captured["worker_env"]["RLLM_NODE_LOCAL_RANK"] == "3"
    assert "CUDA_VISIBLE_DEVICES" not in captured["worker_env"]
    assert captured["worker_env"]["RLLM_WORKER_CUDA_VISIBLE_DEVICES"] == "6"


def test_verl_ray_worker_patch_preserves_ray_owned_cuda_visible_devices(monkeypatch):
    """Ray must not see a single-device actor env before placement id translation."""
    from verl.single_controller.ray import RayWorkerGroup

    captured = {}

    def fake_create_worker(
        self,
        rank,
        pg_idx,
        pg,
        local_rank,
        resource_pool,
        ray_cls_with_init,
        worker_env,
        detached,
    ):
        captured["worker_env"] = worker_env

    monkeypatch.setattr(RayWorkerGroup, "_create_worker", fake_create_worker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(verl_patch, "_VERL_RAY_LOCAL_RANK_ENV_PATCHED", False)
    verl_patch.patch_verl_ray_worker_local_rank_env()

    class ResourcePool:
        store = [8]

    RayWorkerGroup._create_worker(
        object(),
        rank=7,
        pg_idx=0,
        pg=object(),
        local_rank=7,
        resource_pool=ResourcePool(),
        ray_cls_with_init=object(),
        worker_env={},
        detached=False,
    )

    assert captured["worker_env"]["LOCAL_RANK"] == "7"
    assert captured["worker_env"]["RLLM_WORKER_CUDA_VISIBLE_DEVICES"] == "7"
    assert "CUDA_VISIBLE_DEVICES" not in captured["worker_env"]


def test_worker_cuda_binding_uses_local_rank_without_rewriting_visible_devices(monkeypatch):
    calls = []

    class FakeTorchDevice:
        @staticmethod
        def device_count():
            return 8

        @staticmethod
        def set_device(device):
            calls.append(device)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("RLLM_NODE_LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RLLM_WORKER_CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: FakeTorchDevice)

    worker = type("Worker", (), {"__dict__": {}})()
    verl_patch._bind_current_worker_cuda_device(worker)

    assert calls == [3]
    assert worker.__dict__["_local_rank"] == 3
    assert worker.__dict__["_cuda_visible_devices"] == "3"
    assert "CUDA_VISIBLE_DEVICES" in __import__("os").environ
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_worker_cuda_binding_maps_selected_device_to_existing_visible_index(monkeypatch):
    calls = []

    class FakeTorchDevice:
        @staticmethod
        def device_count():
            return 4

        @staticmethod
        def set_device(device):
            calls.append(device)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("RLLM_NODE_LOCAL_RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("RLLM_WORKER_CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: FakeTorchDevice)

    worker = type("Worker", (), {"__dict__": {}})()
    verl_patch._bind_current_worker_cuda_device(worker)

    assert calls == [2]
    assert worker.__dict__["_local_rank"] == 2
    assert __import__("os").environ["LOCAL_RANK"] == "2"


def test_worker_cuda_binding_uses_rllm_visible_devices_when_cuda_env_is_absent(monkeypatch):
    calls = []

    class FakeTorchDevice:
        @staticmethod
        def device_count():
            return 8

        @staticmethod
        def set_device(device):
            calls.append(device)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("RLLM_CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("RLLM_NODE_LOCAL_RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "4")
    monkeypatch.setenv("RLLM_WORKER_CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: FakeTorchDevice)

    worker = type("Worker", (), {"__dict__": {}})()
    verl_patch._bind_current_worker_cuda_device(worker)

    assert calls == [4]
    assert worker.__dict__["_local_rank"] == 4


def test_vllm_server_runtime_env_preserves_worker_setup_hook():
    runtime_env = verl_patch._vllm_server_runtime_env(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
        }
    )

    assert runtime_env["env_vars"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert runtime_env["worker_process_setup_hook"] == "rllm.experimental.verl.patch.apply_all_verl_patches"
