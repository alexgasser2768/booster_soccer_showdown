import torch

from ..learning.agent import Agent
from .utils import create_input_vector
from .simulation import SimulationEnvironment


def visualize(weight_path: str, n_states: int, n_actions: int):
    simulation = SimulationEnvironment()

    agent = Agent(n_states=n_states, n_actions=n_actions)
    agent.loadWeights(weight_path)

    while True:
        # Reset environment for new episode
        terminated, truncated = False, False
        observation, info = simulation.reset()

        while not (terminated or truncated):
            model_input = torch.tensor(create_input_vector(info, observation[:24]).reshape(1, -1), dtype=torch.float32)

            ctrl = agent(model_input)[0]
            ctrl = ctrl.detach().numpy().reshape((12, ))

            observation, reward, terminated, truncated, info = simulation.step(ctrl)

            if terminated or truncated:
                break

        # Check if user exited the simulation
        if simulation.exited:
            simulation.close()
            return