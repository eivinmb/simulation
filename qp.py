import numpy as np


def qp(u_hat, h, grad_h, gamma, max_speed):
    # Return the admissible input closest to the nominal one, and whether an
    # admissible input exists at all.
    #
    #   min ||u - u_hat||^2   s.t.   grad_h . u >= -gamma * h,  ||u||_inf <= ubar
    #
    # For kinematics f = 0 and g = I, so L_f h = 0 and L_g h = grad_h, and the
    # CBF condition is a single half-space. Intersected with the box it is no
    # longer a plain projection, but it is still exact without a solver.
    #
    # The KKT conditions give u_i = clamp(u_hat_i + lam * a_i) for a multiplier
    # lam >= 0, so the whole problem is one scalar unknown. And since every
    # clamped component contributes nothing to the derivative, a . u(lam) is
    # non-decreasing in lam: bisection cannot miss the root.
    a = grad_h
    b = -gamma * h

    def reached(lam):
        return a @ np.clip(u_hat + lam * a, -max_speed, max_speed)

    # Saturating every component in the direction that helps is the most the
    # constraint can ever be pushed. If that is not enough, nothing is, and
    # there is no safe action at all. This is also what catches grad_h = 0,
    # where the row loses all contact with the input.
    if max_speed * np.abs(a).sum() < b:
        return np.clip(u_hat, -max_speed, max_speed), False

    if reached(0.0) >= b:
        # The nominal input already satisfies the row: the filter is inactive.
        return np.clip(u_hat, -max_speed, max_speed), True

    high = 1.0
    while reached(high) < b:
        high *= 2.0

    low = 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if reached(middle) < b:
            low = middle
        else:
            high = middle

    return np.clip(u_hat + high * a, -max_speed, max_speed), True


if __name__ == "__main__":
    # The hare's barrier, h = x^2 - 4, with gamma = 100, a nominal input of 2
    # to the left, and a speed limit of 2.
    gamma, max_speed = 100.0, 2.0
    u_hat = np.array([-2.0, 0.0])
    for x in (5.0, 2.1, 2.0, 1.0, 0.0):
        h = x**2 - 4.0
        grad_h = np.array([2.0 * x, 0.0])
        u, feasible = qp(u_hat, h, grad_h, gamma, max_speed)
        print(
            f"x={x:+5.2f}  h={h:+7.2f}  grad_h={grad_h[0]:+6.2f}  "
            f"u=[{u[0]:+6.3f},{u[1]:+5.2f}]  feasible={feasible}"
        )

    # With the box wide open the answer must be the plain half-space
    # projection, which has nothing in it to tune. That formula stays the
    # oracle: anything that replaces this function has to reproduce it.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(1000):
        nominal, a = rng.standard_normal(2), rng.standard_normal(2)
        value = rng.standard_normal()
        b = -gamma * value
        u, _ = qp(nominal, value, a, gamma, 1e6)
        exact = nominal
        if a @ nominal < b:
            exact = nominal + a * (b - a @ nominal) / (a @ a)
        worst = max(worst, np.max(np.abs(u - exact)))
    print(f"box inactive vs closed-form projection: max difference {worst:.1e}")
