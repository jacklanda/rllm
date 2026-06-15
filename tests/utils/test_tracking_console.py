from rllm.utils.tracking import concat_dict_to_str


def test_console_log_includes_rollout_step_metrics():
    line = concat_dict_to_str(
        {
            "traj/accept_rate": 1.0,
            "rollout_step/elapsed_time": 12.5,
            "rollout_step/total_trajectories": 128.0,
            "rollout_step/completed_trajectories": 128.0,
            "rollout_step/throughput_traj_per_second": 10.24,
            "rollout_step/tail_guard_enabled": 1.0,
            "rollout_step/tail_phase_start_elapsed_seconds": 9.0,
            "rollout_step/tail_guard_early_stop": 3.0,
            "rollout_step/tail_guard_observed_overlong": 12.0,
            "rollout_step/tail_guard_overlong_burst": 1.0,
            "rollout_step/tail_guard_overlong_burst_elapsed_seconds": 10.0,
            "rollout_step/tail_guard_recent_overlong_ratio": 0.75,
            "non_numeric": "not printed",
        },
        step=1,
    )

    assert "step:1" in line
    assert "traj/accept_rate:1.0" in line
    assert "rollout_step/elapsed_time:12.5" in line
    assert "rollout_step/total_trajectories:128.0" in line
    assert "rollout_step/completed_trajectories:128.0" in line
    assert "rollout_step/throughput_traj_per_second:10.24" in line
    assert "rollout_step/tail_guard_enabled:1.0" in line
    assert "rollout_step/tail_phase_start_elapsed_seconds:9.0" in line
    assert "rollout_step/tail_guard_early_stop:3.0" in line
    assert "rollout_step/tail_guard_observed_overlong:12.0" in line
    assert "rollout_step/tail_guard_overlong_burst:1.0" in line
    assert "rollout_step/tail_guard_overlong_burst_elapsed_seconds:10.0" in line
    assert "rollout_step/tail_guard_recent_overlong_ratio:0.75" in line
    assert "non_numeric" not in line
