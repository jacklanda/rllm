from rllm.experimental.common.visualization import visualize_trajectory_last_steps
from rllm.types import Step, Trajectory, TrajectoryGroup

import re


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _FakeTokenizer:
    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)


def _group_with_step(prompt: str, response: str) -> list[TrajectoryGroup]:
    step = Step(
        prompt_ids=[ord(ch) for ch in prompt],
        response_ids=[ord(ch) for ch in response],
        reward=1.0,
    )
    trajectory = Trajectory(name="default_traj_name", steps=[step])
    return [TrajectoryGroup(trajectories=[trajectory], group_id="task-1", metadata=[{}])]


def test_visualize_trajectory_last_steps_prints_full_text_by_default(capsys):
    prompt = "prompt-" + ("abc " * 300) + "prompt-tail"
    response = "response-" + ("xyz " * 300) + "response-tail"

    visualize_trajectory_last_steps(
        _group_with_step(prompt, response),
        tokenizer=_FakeTokenizer(),
        max_steps_to_visualize=1,
        show_workflow_metadata=False,
    )

    output = capsys.readouterr().out
    plain_output = _ANSI_RE.sub("", output)
    assert "skipped middle" not in output
    assert "prompt-tail" in plain_output
    assert "response-tail" in plain_output


def test_visualize_trajectory_last_steps_can_still_abbreviate_when_requested(capsys):
    prompt = "prompt-" + ("abc " * 300) + "prompt-tail"
    response = "response-" + ("xyz " * 300) + "response-tail"

    visualize_trajectory_last_steps(
        _group_with_step(prompt, response),
        tokenizer=_FakeTokenizer(),
        max_steps_to_visualize=1,
        show_workflow_metadata=False,
        max_display_chars=128,
    )

    output = capsys.readouterr().out
    assert "skipped middle" in output
