from ..learning import Agent
from ..environments import Environment


def visualize(simulation: Environment, weight_path: str, n_states: int, n_actions: int):
    agent = Agent(n_states=n_states, n_actions=n_actions)
    agent.loadWeights(weight_path)

    while True:
        # Reset environment for new episode
        terminated, truncated = False, False
        observation = simulation.reset()

        while not (terminated or truncated):
            ctrl = agent(observation)
            ctrl = ctrl.detach().numpy().reshape((12, ))

            observation, _, terminated, truncated = simulation.step(ctrl)

            if terminated or truncated:
                break

        # Check if user exited the simulation
        if simulation.is_closed:
            simulation.close()
            return