import numpy as np
import sympy

COORDINATES = ("x", "y")


def environment_constants(environment, prefix=""):
    # Flatten the environment into names an expression can use, so the geometry
    # has one home: {"water": {"center": 0, "width": 4}} becomes water_center
    # and water_width.
    constants = {}
    for key, value in environment.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            constants.update(environment_constants(value, prefix=f"{name}_"))
        elif isinstance(value, (int, float)):
            constants[name] = float(value)
    return constants


def make_barrier(expression, environment):
    # A barrier is written in the setup file as an expression in the position
    # coordinates plus any constant from the environment, and differentiated
    # symbolically. A new barrier therefore needs no code at all:
    #
    #   "(x - water_center)**2 - (water_width/2)**2"   the paper's hare
    #   "0"                                            the identity CBF
    #
    # Safety is h >= 0, so the expression is positive where the agent is safe.
    symbols = sympy.symbols(COORDINATES)
    expr = sympy.sympify(str(expression)).subs(environment_constants(environment))

    unknown = sorted(str(s) for s in expr.free_symbols if s not in symbols)
    if unknown:
        raise ValueError(f"barrier {expression!r} uses unknown names {unknown}")

    h_func = sympy.lambdify(symbols, expr, "numpy")
    grad_func = sympy.lambdify(symbols, [sympy.diff(expr, s) for s in symbols], "numpy")

    def h(position):
        return float(h_func(*position))

    def grad_h(position):
        return np.array(grad_func(*position), dtype=float)

    return h, grad_h


def gradient_error(h, grad_h, position, eps=1e-6):
    # Central difference against the symbolic gradient. The derivative itself
    # is exact; this checks that substitution and lambdify did what they look
    # like they did.
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
    print("constants", environment_constants(setup_data["environment"]))
    for name in setup_data["agents"]:
        expression = setup_data[name]["barrier"]
        h, grad_h = make_barrier(expression, setup_data["environment"])
        worst = max(gradient_error(h, grad_h, p) for p in probes)
        values = "  ".join(f"h({p[0]:+.0f})={h(p):+6.1f}" for p in probes)
        print(f"{name:7} {values}   max gradient error {worst:.1e}   {expression!r}")
