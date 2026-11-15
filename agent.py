import numpy as np

import kinematics


class Agent:
    def __init__(self, name, position, goal_position, max_speed, barrier):
        self.name = name
        self.position = np.array(position, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)
        self.max_speed = max_speed
        self._h, self._grad_h = barrier

    # The velocity is the input, not a state, so it comes in from the outside
    # every step. Nothing stale can be left lying on the agent.
    def step(self, velocity, dt):
        self.position = kinematics.linear_kinematics_update(
            self.position, velocity, dt
        )

    # The dynamics themselves stay in kinematics.py as plain functions.
    def f(self):
        return kinematics.kinematics_f(self.position)

    def g(self):
        return kinematics.kinematics_g(self.position)

    # The barrier is bound to the environment when the agent is loaded, so
    # the agent only has to supply its own position.
    def h(self):
        return self._h(self.position)

    def grad_h(self):
        return self._grad_h(self.position)

    def __repr__(self):
        return f"Agent({self.name}, position={self.position})"
