import numpy as np


def linear_kinematics_update(position, velocity, dt):
    new_position = position + velocity * dt
    return new_position


# The same kinematics written as xdot = f + g u, which is the form the CBF-QP
# needs: the barrier constraint grad(h) . xdot >= -alpha(h) is only linear in u
# if f and g are available separately. For kinematics the velocity *is* the
# input, so the drift is zero and the input matrix is the identity.
def kinematics_f(position):
    return np.zeros_like(position, dtype=float)


def kinematics_g(position):
    return np.eye(len(position))
