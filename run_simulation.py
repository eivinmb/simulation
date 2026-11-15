import sys

import numpy as np

from qp import qp
from setup import initialize_simulation

DEFAULT_SETUP = "egersted_paperet/egersted.json"


def nominal_velocity(agent, gain):
    # Proportional pull towards the goal, saturated at the agent's speed limit.
    error = agent.goal_position - agent.position
    velocity = gain * error
    speed = np.linalg.norm(velocity)
    if speed > agent.max_speed:
        velocity = velocity * agent.max_speed / speed
    return velocity


def filtered_velocity(agent, gain, gamma):
    # The nominal controller wants the goal and ignores safety. The QP hands
    # back the closest input that keeps h from falling faster than allowed.
    return qp(nominal_velocity(agent, gain), agent.h(), agent.grad_h(), gamma)


def run(setup_file):
    agents, setup_data = initialize_simulation(setup_file)
    simulation = setup_data["simulation"]
    dt = simulation["dt"]
    gain = simulation["gain"]
    gamma = simulation["gamma"]
    steps = int(simulation["duration"] / dt)

    time = np.arange(steps + 1) * dt
    positions = {agent.name: np.zeros((steps + 1, 2)) for agent in agents}
    infeasible = {agent.name: 0 for agent in agents}
    for agent in agents:
        positions[agent.name][0] = agent.position

    for k in range(steps):
        for agent in agents:
            velocity, feasible = filtered_velocity(agent, gain, gamma)
            infeasible[agent.name] += not feasible
            agent.step(velocity, dt)
            positions[agent.name][k + 1] = agent.position

    return time, positions, infeasible, setup_data


if __name__ == "__main__":
    setup_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETUP
    time, positions, infeasible, setup_data = run(setup_file)
    for name, track in positions.items():
        print(
            f"{name:7} start {track[0]}  end {track[-1]}  "
            f"infeasible steps {infeasible[name]}"
        )
