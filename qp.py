import numpy as np


def qp(u_hat, h, grad_h, gamma):
    # Return the admissible input closest to the nominal one, and whether an
    # admissible input exists at all.
    #
    # For kinematics f = 0 and g = I, so L_f h = 0 and L_g h = grad_h, and the
    # CBF condition grad_h . u >= -gamma * h is a single half-space. The
    # closest point in a half-space is the projection onto its boundary, so no
    # solver is needed: the answer is exact and there is nothing to tune.
    a = grad_h
    b = -gamma * h

    if a @ a == 0.0:
        # The gradient vanishes, so no input changes h at all and the row reads
        # 0 >= b. Satisfied when h >= 0, infeasible otherwise. This is not an
        # edge case to paper over: it happens wherever the barrier has an
        # interior extremum, which for the water strip is its centre line.
        return u_hat, b <= 0.0

    if a @ u_hat >= b:
        # The nominal input already satisfies the condition, so the filter is
        # inactive and passes it through untouched.
        return u_hat, True

    return u_hat + a * (b - a @ u_hat) / (a @ a), True


if __name__ == "__main__":
    # The hare's barrier, h = x^2 - 4, with gamma = 100 and a nominal input of
    # 2 to the left. Checked against the numbers worked out by hand.
    gamma = 100.0
    u_hat = np.array([-2.0, 0.0])
    for x in (5.0, 2.1, 2.0, 1.0, 0.0):
        h = x**2 - 4.0
        grad_h = np.array([2.0 * x, 0.0])
        u, feasible = qp(u_hat, h, grad_h, gamma)
        print(
            f"x={x:+5.2f}  h={h:+7.2f}  grad_h={grad_h[0]:+6.2f}  "
            f"u=[{u[0]:+6.3f},{u[1]:+5.2f}]  feasible={feasible}"
        )
