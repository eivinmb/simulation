import numpy as np

from setup import initialize_simulation


def nominal_velocity(agent, gain=1.0):
    # Proportional pull towards the goal, saturated at the agent's speed limit.
    error = agent.goal_position - agent.position
    velocity = gain * error
    speed = np.linalg.norm(velocity)
    if speed > agent.max_speed:
        velocity = velocity * agent.max_speed / speed
    return velocity


def run(setup_file, dt=0.01, duration=20.0):
    agents, environment = initialize_simulation(setup_file)
    steps = int(duration / dt)

    time = np.arange(steps + 1) * dt
    positions = {agent.name: np.zeros((steps + 1, 2)) for agent in agents}
    for agent in agents:
        positions[agent.name][0] = agent.position

    for k in range(steps):
        for agent in agents:
            agent.step(nominal_velocity(agent), dt)
            positions[agent.name][k + 1] = agent.position

    return time, positions, environment


if __name__ == "__main__":
    time, positions, environment = run("egersted_paperet/egersted.json")
    for name, track in positions.items():
        print(name, "start", track[0], "end", track[-1])
