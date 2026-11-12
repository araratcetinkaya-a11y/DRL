"""Characterization tests: compute_reward_final gives the same rewards as the thesis code.

The values were recorded by running the frozen thesis code (tag thesis-final) on the fake AirSim
client before compute_reward_final was changed to use gym_env/core/reward.py.

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
    ((1.0, 0.0), 10.0, [-0.0049517139873444214, 0.02004828601265558, 0.0, 0.0, 0.025, 0.0], False),
    ((1.0, 0.0), 10.0, [0.036011330648506146, 0.036011330648506146, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.05184433807064011, 0.05184433807064011, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.06748848079512867, 0.06748848079512867, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.08288380107955504, 0.08288380107955504, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.09796885862099884, 0.09796885862099884, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.11268037528772083, 0.11268037528772083, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.12695287699905294, 0.12695287699905294, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.14071833343662093, 0.14071833343662093, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.0), 10.0, [0.15390579700279544, 0.15390579700279544, 0.0, 0.0, 0.0, 0.0], False),
    ((1.0, 0.5), 10.0, [0.15894104336472195, 0.16644104336472196, 0.0, 0.0, 0.0, 0.0075], False),
    ((0.0, -1.0), 10.0, [0.054277206280125945, 0.14677720628012594, 0.0, 0.0, 0.025, 0.0675], False),
    ((-1.0, 1.0), 10.0, [0.01705295354270675, 0.16205295354270674, 0.0, 0.0, 0.025, 0.12], False),
    ((1.0, 0.0), 1.5, [-75.11346320821889, 0.14153679178112383, 0.125, 50.0, 0.1, 0.03], True),
]

# total reward, distance term (each case starts from a fresh, seeded environment)
TERMINAL = {
    "reach": (69.6, 39.6),      # 0.5 m from the goal: 40 * 49.5 / 50 + 30
    "crash": (-25.0, 0.0),
    "outside": (-88.74904713883294, -78.74904713883294),
}


def test_scripted_episode_rewards_match_thesis_code(env, client):
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
def test_terminal_rewards_match_thesis_code(thesis_cfg, case, setup):
    environment = build_env(thesis_cfg)
    environment.reset()
    raw = environment.unwrapped
    setup(raw, raw.dynamic_model.client)

    raw.get_obs()  # updates min_distance_to_obstacles, as in step()
    reward, terms, _ = raw.compute_reward_final(True, np.array([0.0, 0.0], dtype=np.float32), return_terms=True)

    expected_total, expected_distance = TERMINAL[case]
    assert float(reward) == pytest.approx(expected_total, rel=1e-6)
    assert float(terms["r_dist"]) == pytest.approx(expected_distance, rel=1e-6, abs=1e-9)
