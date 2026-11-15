import sys

import matplotlib.pyplot as plt
import numpy as np

from run_simulation import DEFAULT_SETUP, nominal_velocity
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


def animate(setup_file):
    agents, setup_data = initialize_simulation(setup_file)
    simulation = setup_data["simulation"]
    dt = simulation["dt"]
    fps = simulation["fps"]
    gain = simulation["gain"]

    fig, (ax, ax_h) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[2, 1], constrained_layout=True
    )
    draw_environment(ax, setup_data["environment"])
    corners = [agent.position for agent in agents]
    corners += [agent.goal_position for agent in agents]
    corners = np.array(corners)
    ax.set_xlim(corners[:, 0].min() - 1, corners[:, 0].max() + 1)
    ax.set_ylim(corners[:, 1].min() - 1, corners[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(setup_data["title"])

    # h is the whole point: positive is safe, negative is not, and the zero
    # line is the boundary of the safe set.
    ax_h.axhline(0.0, color="k", lw=0.8)
    ax_h.set_xlim(0.0, simulation["duration"])
    ax_h.set_xlabel("t [s]")
    ax_h.set_ylabel("h")

    trails = {}
    markers = {}
    tracks = {}
    curves = {}
    values = {}
    for agent in agents:
        (markers[agent.name],) = ax.plot([], [], "o", label=agent.name)
        color = markers[agent.name].get_color()
        (trails[agent.name],) = ax.plot([], [], "-", lw=1, color=color)
        ax.plot(*agent.goal_position, "x", color=color)
        tracks[agent.name] = [agent.position.copy()]
        (curves[agent.name],) = ax_h.plot([], [], "-", lw=1.2, color=color)
        values[agent.name] = [agent.h()]
    ax.legend(loc="best")
    time_text = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right")

    # The physics step is smaller than a frame, so take several steps per frame
    # and pause for as long as those steps represent. That keeps the animation
    # close to real time without tying the integration to the screen refresh.
    steps_per_frame = max(1, round(1.0 / (fps * dt)))
    frames = int(simulation["duration"] / (dt * steps_per_frame))

    t = 0.0
    times = [t]
    for _ in range(frames):
        for _ in range(steps_per_frame):
            for agent in agents:
                agent.step(nominal_velocity(agent, gain), dt)
                tracks[agent.name].append(agent.position.copy())
        t += steps_per_frame * dt
        times.append(t)
        for agent in agents:
            track = np.array(tracks[agent.name])
            trails[agent.name].set_data(track[:, 0], track[:, 1])
            markers[agent.name].set_data([agent.position[0]], [agent.position[1]])
            values[agent.name].append(agent.h())
            curves[agent.name].set_data(times, values[agent.name])
        ax_h.relim()
        ax_h.autoscale_view(scalex=False)
        time_text.set_text(f"t = {t:.1f} s")
        plt.pause(steps_per_frame * dt)

    plt.show()


if __name__ == "__main__":
    animate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETUP)
