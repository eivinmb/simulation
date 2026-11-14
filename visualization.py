import matplotlib.pyplot as plt
import numpy as np

from run_simulation import nominal_velocity
from setup import initialize_simulation


def draw_environment(ax, environment):
    water = environment["water"]
    ax.add_patch(
        plt.Rectangle(
            (water["x"], water["y"]),
            water["width"],
            water["height"],
            color="tab:blue",
            alpha=0.3,
        )
    )
    return water


def animate(setup_file, dt=0.01, duration=20.0, fps=30):
    agents, environment = initialize_simulation(setup_file)

    fig, ax = plt.subplots()
    water = draw_environment(ax, environment)
    ax.set_xlim(min(0.0, water["x"]) - 1, water["x"] + water["width"] + 1)
    ax.set_ylim(water["y"] - 1, water["y"] + water["height"] + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    trails = {}
    markers = {}
    tracks = {}
    for agent in agents:
        (markers[agent.name],) = ax.plot([], [], "o", label=agent.name)
        color = markers[agent.name].get_color()
        (trails[agent.name],) = ax.plot([], [], "-", lw=1, color=color)
        ax.plot(*agent.goal_position, "x", color=color)
        tracks[agent.name] = [agent.position.copy()]
    ax.legend(loc="upper left")

    # The physics step is smaller than a frame, so take several steps per frame
    # and pause for as long as those steps represent. That keeps the animation
    # close to real time without tying the integration to the screen refresh.
    steps_per_frame = max(1, round(1.0 / (fps * dt)))
    frames = int(duration / (dt * steps_per_frame))

    for _ in range(frames):
        for _ in range(steps_per_frame):
            for agent in agents:
                agent.step(nominal_velocity(agent), dt)
                tracks[agent.name].append(agent.position.copy())
        for agent in agents:
            track = np.array(tracks[agent.name])
            trails[agent.name].set_data(track[:, 0], track[:, 1])
            markers[agent.name].set_data([agent.position[0]], [agent.position[1]])
        plt.pause(steps_per_frame * dt)

    plt.show()


if __name__ == "__main__":
    animate("egersted_paperet/egersted.json")
