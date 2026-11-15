import numpy as np
import sympy

COORDINATES = ("x", "y")


def strip(region, symbols):
    # A band of the given width centred on "center" along "axis". The paper's
    # water: h = (p_axis - c)^2 - (l_R/2)^2, positive outside the band.
    axis = symbols[COORDINATES.index(region.get("axis", "x"))]
    return (axis - region["center"]) ** 2 - (region["width"] / 2) ** 2


def disc(region, symbols):
    # A circle of the given radius about "center". Positive outside it.
    squared = sum((s - c) ** 2 for s, c in zip(symbols, region["center"]))
    return squared - region["radius"] ** 2


SHAPES = {"strip": strip, "disc": disc}


def region_expression(name, environment, symbols):
    if name not in environment:
        raise ValueError(f"unknown region {name!r}, expected one of {sorted(environment)}")
    region = environment[name]
    shape = region["shape"]
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}, expected one of {sorted(SHAPES)}")
    return SHAPES[shape](region, symbols)


def make_barrier(unsafe_in, environment):
    # An agent's barrier is built from the regions it cannot enter. Each region
    # contributes an expression that is positive outside it, and the agent is
    # safe only outside every one of them, so the barrier is their minimum.
    #
    # Minimum is conjunction. The paper rejects it for *collaborative* terms,
    # because a conjunction can never expand a safe set, but for independent
    # hazards it is the right operator: two dangers must not cancel each other.
    # It is not smooth, so the gradient below is the active region's -- the
    # usual subgradient choice, and exactly the non-smoothness the paper cites.
    symbols = sympy.symbols(COORDINATES)
    regions = []
    for name in unsafe_in:
        expr = region_expression(name, environment, symbols)
        regions.append(
            (
                sympy.lambdify(symbols, expr, "numpy"),
                sympy.lambdify(symbols, [sympy.diff(expr, s) for s in symbols], "numpy"),
            )
        )

    def h(position):
        # No regions means safe everywhere: the paper's identity CBF, h = 0.
        # Zero and not one, because safety is non-negativity, not positivity,
        # and zero is the identity element for addition.
        if not regions:
            return 0.0
        return min(float(value(*position)) for value, _ in regions)

    def grad_h(position):
        if not regions:
            return np.zeros(len(COORDINATES))
        values = [float(value(*position)) for value, _ in regions]
        active = int(np.argmin(values))
        return np.array(regions[active][1](*position), dtype=float)

    return h, grad_h


def gradient_error(h, grad_h, position, eps=1e-6):
    # Central difference against the symbolic gradient. The derivative itself
    # is exact; this checks that lambdify did what it looks like it did.
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
        unsafe_in = setup_data[name]["unsafe_in"]
        h, grad_h = make_barrier(unsafe_in, setup_data["environment"])
        worst = max(gradient_error(h, grad_h, p) for p in probes)
        values = "  ".join(f"h({p[0]:+.0f})={h(p):+6.1f}" for p in probes)
        print(f"{name:7} {values}   max gradient error {worst:.1e}   unsafe_in={unsafe_in}")
