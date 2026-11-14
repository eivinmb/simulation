import matplotlib.pyplot as plt
import numpy as np

from run_simulation import nominal_velocity
from setup import initialize_simulation


def draw_environment(ax, environment):
    # The water is a strip of width l_R centred on the origin, as in the paper.
    # It spans the whole height of the plot, so y never enters the geometry.
    water = environment["water"]
    half_width = water["width"] / 2
    ax.axvspan(
        water["center"] - half_width,
        water["center"] + half_width,
        color="tab:blue",
        alpha=0.3,
    )


def animate(
    setup_file, dt=0.01, duration=20.0, fps=30, title="Ground-amphibious robots"
):
    agents, environment = initialize_simulation(setup_file)

    fig, ax = plt.subplots()
    draw_environment(ax, environment)
    corners = [agent.position for agent in agents]
    corners += [agent.goal_position for agent in agents]
    corners = np.array(corners)
    ax.set_xlim(corners[:, 0].min() - 1, corners[:, 0].max() + 1)
    ax.set_ylim(corners[:, 1].min() - 1, corners[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    trails = {}
    markers = {}
    tracks = {}
    for agent in agents:
        (markers[agent.name],) = ax.plot([], [], "o", label=agent.name)
        color = markers[agent.name].get_color()
        (trails[agent.name],) = ax.plot([], [], "-", lw=1, color=color)
        ax.plot(*agent.goal_position, "x", color=color)
        tracks[agent.name] = [agent.position.copy()]
    ax.legend(loc="best")
    time_text = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right")

    # The physics step is smaller than a frame, so take several steps per frame
    # and pause for as long as those steps represent. That keeps the animation
    # close to real time without tying the integration to the screen refresh.
    steps_per_frame = max(1, round(1.0 / (fps * dt)))
    frames = int(duration / (dt * steps_per_frame))

    t = 0.0
    for _ in range(frames):
        for _ in range(steps_per_frame):
            for agent in agents:
                agent.step(nominal_velocity(agent), dt)
                tracks[agent.name].append(agent.position.copy())
        for agent in agents:
            track = np.array(tracks[agent.name])
            trails[agent.name].set_data(track[:, 0], track[:, 1])
            markers[agent.name].set_data([agent.position[0]], [agent.position[1]])
        t += steps_per_frame * dt
        time_text.set_text(f"t = {t:.1f} s")
        plt.pause(steps_per_frame * dt)

    plt.show()


if __name__ == "__main__":
    animate("egersted_paperet/egersted.json")
