from rllm.experimental.verl.patch import patch_verl_ray_worker_local_rank_env


def test_verl_ray_worker_patch_sets_standard_local_rank_env(monkeypatch):
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
    patch_verl_ray_worker_local_rank_env()

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
