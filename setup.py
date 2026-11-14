import json

from agent import Agent


def read_setup_file(setup_file):
    with open(setup_file, "r") as f:
        setup_data = json.load(f)
    return setup_data


def load_agents(setup_data):
    agents = []
    for name in setup_data["agents"]:
        spec = setup_data[name]
        # initial_velocity is read as the agent's speed limit: the hare is
        # faster than the turtle, and that difference is the heterogeneity.
        agents.append(
            Agent(
                name=name,
                position=spec["initial_position"],
                goal_position=spec["goal_position"],
                max_speed=spec["initial_velocity"],
            )
        )
    return agents


def initialize_simulation(setup_file):
    setup_data = read_setup_file(setup_file)
    agents = load_agents(setup_data)
    environment = setup_data["environment"]
    return agents, environment
