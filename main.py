import yaml, time, logging, os
import numpy as np
from torchrl.envs import check_env_specs

from src.environments import Environment, WalkToBallEnv, PenaltyKickEnv, GoaliePenaltyKickEnv, ObstaclePenaltyKickEnv, KickToTargetEnv
from src.utils import teleop, visualize
from src.learning import behavior_cloning

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_directory = config['dataset_directory']
    weights_directory = config['weights_directory']
    logging_directory = config['logging_directory']

    for directory in [dataset_directory, weights_directory, logging_directory]:
        os.makedirs(directory, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=f"{logging_directory}/{time.time()}.log", level=logging.INFO)
    logger.info('Started')

    n_states = config['model']['states']
    n_actions = config['model']['actions']

    env_name = config['environment']['name']
    env_headless = config['environment']['headless']
    env_max_episode_steps = config['environment']['max_episode_steps']

    simulation = Environment(env_name=env_name, headless=env_headless, max_episodes=env_max_episode_steps)

    if config['teleop']['enabled']:
        logger.info("Starting Teleoperation Data Collection...")

        file_prefix = config['teleop']['file_prefix']
        pos_sensitivity = config['teleop']['position_sensitivity']
        rot_sensitivity = config['teleop']['rotation_sensitivity']

        dataset = teleop(
            simulation=simulation,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity
        )

        # Save the collected dataset
        dataset_path = f"{dataset_directory}/{file_prefix}-{time.time()}.npz"
        np.savez_compressed(
            dataset_path,
            observations=np.array(dataset["observations"]),
            infos=np.array(dataset["infos"]),
            actions=np.array(dataset["actions"])
        )

        logger.info(f"Dataset saved to {dataset_path}")

    if config["visualize"]["enabled"]:
        logger.info("Starting Model Visualization...")

        weight_path = f"{weights_directory}/{config['visualize']['weight_file']}"

        visualize(
            simulation=simulation,
            weight_path=weight_path,
            n_states=n_states,
            n_actions=n_actions
        )

    if config['behavior_cloning']['enabled']:
        logger.info("Starting Behavior Cloning Training...")

        data_files = config['behavior_cloning']['data_files']
        batch_size = config['behavior_cloning']['batch_size']
        epochs = config['behavior_cloning']['epochs']
        learning_rate = config['behavior_cloning']['learning_rate']
        file_prefix = config['behavior_cloning']['file_prefix']

        behavior_cloning(
            data_files=[f"{dataset_directory}/{data_file}" for data_file in data_files],
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            n_states=n_states,
            n_actions=n_actions,
            model_weights_directory=weights_directory,
            model_weights_prefix=file_prefix
        )

    logger.info('Finished')