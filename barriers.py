import numpy as np


def water_strip(water):
    # The hare is safe on land and unsafe in the water. The strip is centred on
    # water["center"] with width water["width"], so h is the squared distance
    # from the centre line minus the squared half width: the paper's
    # h_r = p_x^2 - (l_R/2)^2. Positive on land, negative in the water, and
    # independent of y.
    center = water["center"]
    half_width = water["width"] / 2

    def h(position):
        return (position[0] - center) ** 2 - half_width**2

    def grad_h(position):
        gradient = np.zeros_like(position)
        gradient[0] = 2 * (position[0] - center)
        return gradient

    return h, grad_h


def no_barrier(water):
    # The amphibious agent is safe everywhere: the paper's identity CBF. Note
    # that it is zero and not one. Safety requires non-negativity rather than
    # positivity, and zero is the identity element for addition, which is the
    # composition operator the paper settles on.
    def h(position):
        return 0.0

    def grad_h(position):
        return np.zeros_like(position)

    return h, grad_h


BARRIERS = {"water_strip": water_strip, "none": no_barrier}


def make_barrier(name, environment):
    if name not in BARRIERS:
        raise ValueError(f"unknown barrier {name!r}, expected one of {list(BARRIERS)}")
    return BARRIERS[name](environment["water"])


def gradient_error(h, grad_h, position, eps=1e-6):
    # Central difference against the analytic gradient. A sign slip in a
    # gradient gives a simulation that looks plausible and proves nothing, so
    # every barrier is checked before it is used.
    position = np.asarray(position, dtype=float)
    numerical = np.zeros_like(position)
    for i in range(position.size):
        step = np.zeros_like(position)
        step[i] = eps
        numerical[i] = (h(position + step) - h(position - step)) / (2 * eps)
    return np.max(np.abs(numerical - grad_h(position)))


if __name__ == "__main__":
    import json
    import sys

    setup_file = sys.argv[1] if len(sys.argv) > 1 else "egersted_paperet/egersted.json"
    with open(setup_file, "r") as f:
        setup_data = json.load(f)

    probes = [[-5.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [2.0, 0.0], [3.0, 1.0]]
    for name in setup_data["agents"]:
        h, grad_h = make_barrier(setup_data[name]["barrier"], setup_data["environment"])
        worst = max(gradient_error(h, grad_h, p) for p in probes)
        values = "  ".join(f"h({p[0]:+.0f})={h(np.array(p)):+6.1f}" for p in probes)
        print(f"{name:7} {values}   max gradient error {worst:.2e}")
