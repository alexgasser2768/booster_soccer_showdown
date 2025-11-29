import yaml, time, logging, os
import numpy as np

from src.environments import Environment, WalkToBallEnv, PenaltyKickEnv, GoaliePenaltyKickEnv, ObstaclePenaltyKickEnv, KickToTargetEnv
from src.utils import teleop, visualize
from src.learning import behavior_cloning, PPOTrainer

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # env = WalkToBallEnv(headless=True)
    # trainer = PPOTrainer(
    #     env=env,
    #     n_states=52,
    #     n_actions=12,
    #     lr=0.0001,
    #     max_grad_norm = 1.0,
    #     frames_per_batch = 1000,
    #     total_frames = 50_000,
    #     sub_batch_size = 64,
    #     num_epochs = 10,
    #     clip_epsilon=0.2,
    #     gamma = 0.99,
    #     lmbda = 0.95,
    #     entropy_eps = 1e-4,
    #     weight_dir="weights/",
    #     weight_file="behavior_cloning-1764439282.0633442.pt",
    #     prefix="PPO"
    # )

    # trainer.train()

    # exit()

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

    if config['ppo']['enabled']:
        logger.info("Starting PPO Training...")

        task = config['ppo']['task']
        epochs = config['ppo']['epochs']
        batch_size = config['ppo']['batch_size']
        clip_epsilon = config['ppo']['clip_epsilon']
        gamma = config['ppo']['gamma']
        lmbda = config['ppo']['lmbda']
        entropy_eps = config['ppo']['entropy_eps']
        value_loss_coeff = config['ppo']['value_loss_coeff']
        max_grad_norm = config['ppo']['max_grad_norm']
        learning_rate = config['ppo']['learning_rate']
        total_frames = config['ppo']['total_frames']
        frames_per_batch = config['ppo']['frames_per_batch']
        weight_file = config['ppo']['weight_file']
        file_prefix = config['ppo']['file_prefix']

        env = WalkToBallEnv()
        if task == "PenaltyKick":
            env = PenaltyKickEnv()
        elif task == "GoaliePenaltyKick":
            env = GoaliePenaltyKickEnv()
        elif task == "ObstaclePenaltyKick":
            env = ObstaclePenaltyKickEnv()
        elif task == "KickToTarget":
            env = KickToTargetEnv()

        trainer = PPOTrainer(
            env = env,
            n_states = n_states,
            n_actions = n_actions,
            lr = learning_rate,
            max_grad_norm = max_grad_norm,
            frames_per_batch = frames_per_batch,
            total_frames = total_frames,
            sub_batch_size = batch_size,
            num_epochs = epochs,
            clip_epsilon = clip_epsilon,
            gamma = gamma,
            lmbda = lmbda,
            entropy_eps = entropy_eps,
            weight_dir = weights_directory,
            weight_file = weight_file,
            prefix = file_prefix
        )

        trainer.train()

    logger.info('Finished')