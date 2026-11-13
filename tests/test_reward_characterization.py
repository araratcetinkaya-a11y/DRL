"""Characterization tests: compute_reward_final gives the same rewards as the training-run code.

The values were recorded by running the code of the thesis training run (tag thesis-final.1,
see docs/TRAINING_RUN.md) on the fake AirSim client.

1. A scripted episode: ten steps accelerating towards an obstacle at 10 m, three steps with
   changing actions (jerk penalties), then the obstacle jumps to 1.5 m at about 4.5 m/s. That last
   step is a crash, and both obstacle penalties are active (Q3 in docs/AUDIT.md).
2. The three terminal cases, each in a fresh environment: reach, crash and leaving the workspace.
"""
import numpy as np
import pytest

from tests.conftest import build_env

KEYS = ["r_total", "r_dist", "c_obs", "c_spd_obs", "c_jerk_fwd", "c_jerk_yaw"]

# (action, obstacle distance [m], [r_total, r_dist, c_obs, c_spd_obs, c_jerk_fwd, c_jerk_yaw], done)
EPISODE = [
    ((1.0, 0.0), 10.0, [6.0357515819474405e-05, 0.025060357515819476, 0.0, 0.0, 0.025, 0.0], False),
    ((1.0, 0.0), 10.0, [0.04501416331063268, 0.04501416331063268, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.06480542258830013, 0.06480542258830013, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.08436060099391085, 0.08436060099391085, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.1036047513494438, 0.1036047513494438, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.12246107327624856, 0.12246107327624856, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.14085046910965104, 0.14085046910965104, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.15869109624881617, 0.15869109624881617, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.17589791679577615, 0.17589791679577615, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.19238224625349432, 0.19238224625349432, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.5), 10.0, [0.20055130420590245, 0.20805130420590245, 0.0, 0.0, 0.0, 0.0075], False),
    ((0.0, -1.0), 10.0, [0.09097150785015742, 0.18347150785015742, 0.0, 0.0, 0.025, 0.0675], False),
    ((-1.0, 1.0), 10.0, [0.05756619192838344, 0.20256619192838343, 0.0, 0.0, 0.025, 0.12], False),
    ((1.0, 0.0), 1.5, [-25.328079010273594, 0.1769209897264048, 0.125, 0.25, 0.1, 0.03], True),
]

# total reward, distance term (each case starts from a fresh, seeded environment)
TERMINAL = {
    "reach": (79.5, 49.5),      # 0.5 m from the goal: 50 * 49.5 / 50 + 30
    "crash": (-25.0, 0.0),
    "outside": (-108.43630892354116, -98.43630892354116),
}


def test_scripted_episode_rewards_match_training_run_code(env, client):
    env.reset()

    for step, (action, obstacle_m, expected_terms, expected_done) in enumerate(EPISODE):
        client.depth_m = obstacle_m
        _, reward, done, info = env.step(np.array(action, dtype=np.float32))

        actual_terms = [float(info["reward_terms"][k]) for k in KEYS]
        assert actual_terms == pytest.approx(expected_terms, rel=1e-6, abs=1e-9), f"step {step}"
        assert float(reward) == pytest.approx(expected_terms[0], rel=1e-6, abs=1e-9), f"step {step}"
        assert bool(done) == expected_done, f"step {step}"


def place_at_goal(raw, client):
    gx, gy, _ = raw.dynamic_model.goal_position  # x and y only: AirSim z is NED (down is positive)
    client._position.x_val, client._position.y_val = gx + 0.5, gy


def collide(raw, client):
    client.has_collided = True


def leave_workspace(raw, client):
    client._position.x_val = 100.0  # SimpleAvoid workspace is +/-60 m around the start


@pytest.mark.parametrize("case, setup", [("reach", place_at_goal), ("crash", collide), ("outside", leave_workspace)])
def test_terminal_rewards_match_training_run_code(thesis_cfg, case, setup):
    environment = build_env(thesis_cfg)
    environment.reset()
    raw = environment.unwrapped
    setup(raw, raw.dynamic_model.client)

    raw.get_obs()  # updates min_distance_to_obstacles, as in step()
    reward, terms, _ = raw.compute_reward_final(True, np.array([0.0, 0.0], dtype=np.float32), return_terms=True)

    expected_total, expected_distance = TERMINAL[case]
    assert float(reward) == pytest.approx(expected_total, rel=1e-6)
    assert float(terms["r_dist"]) == pytest.approx(expected_distance, rel=1e-6, abs=1e-9)
