from ..learning import Agent
from ..environments import Environment


def visualize(simulation: Environment, weight_path: str, n_states: int, n_actions: int):
    agent = Agent(n_states=n_states, n_actions=n_actions)
    agent.loadWeights(weight_path)

    while True:
        # Reset environment for new episode
        terminated, truncated = False, False
        observation, info, model_input = simulation.reset()

        while not (terminated or truncated):
            ctrl = agent(model_input)[0]
            ctrl = ctrl.detach().numpy().reshape((12, ))

            observation, reward, terminated, truncated, info, model_input = simulation.step(ctrl)

            if terminated or truncated:
                break

        # Check if user exited the simulation
        if simulation.exited:
            simulation.close()
            return