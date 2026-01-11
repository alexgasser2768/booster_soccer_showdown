import yaml, time, logging, os
import numpy as np
import matplotlib.pyplot as plt

from src.environments import Environment, WalkToBallEnv, PenaltyKickEnv, GoaliePenaltyKickEnv, ObstaclePenaltyKickEnv, KickToTargetEnv
from src.utils import teleop, visualize
from src.learning import behavior_cloning, PPOTrainer

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_directory = config['dataset_directory']
    weights_directory = config['weights_directory']
    logging_directory = config['logging_directory']

    for directory in [dataset_directory, weights_directory, logging_directory, f"{logging_directory}/plots/"]:
        os.makedirs(directory, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=f"{logging_directory}/{time.time()}.log", level=logging.INFO)
    logger.info('Started')

    n_states = config['model']['states']
    n_actions = config['model']['actions']

    env_name = config['environment']['name']
    env_headless = config['environment']['headless']
    env_max_episode_steps = config['environment']['max_episode_steps']

    #simulation = Environment(env_name=env_name, headless=env_headless, max_episodes=env_max_episode_steps)

    if config['teleop']['enabled']:
        logger.info("Starting Teleoperation Data Collection...")

        file_prefix = config['teleop']['file_prefix']
        pos_sensitivity = config['teleop']['position_sensitivity']
        rot_sensitivity = config['teleop']['rotation_sensitivity']
        simulation = Environment(env_name=env_name, headless=env_headless, max_episodes=env_max_episode_steps)

        dataset = teleop(
            simulation=simulation,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity
        )
        simulation.close()
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
        simulation = Environment(env_name=env_name, headless=env_headless, max_episodes=env_max_episode_steps)

        visualize(
            simulation=simulation,
            weight_path=weight_path,
            n_states=n_states,
            n_actions=n_actions
        )
        simulation.close()

    if config['behavior_cloning']['enabled']:
        logger.info("Starting Behavior Cloning Training...")

        data_files = config['behavior_cloning']['data_files']
        batch_size = config['behavior_cloning']['batch_size']
        epochs = config['behavior_cloning']['epochs']
        learning_rate = config['behavior_cloning']['learning_rate']
        file_prefix = config['behavior_cloning']['file_prefix']
        weight_file = config['behavior_cloning']['weight_file']

        behavior_cloning(
            data_files=[f"{dataset_directory}/{data_file}" for data_file in data_files],
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            n_states=n_states,
            n_actions=n_actions,
            model_weights_directory=weights_directory,
            model_weights_prefix=file_prefix,
            model_weights_file=weight_file,
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
        data = trainer.train()

        if data is not None:
            plt.figure(figsize=(15, 25)) 

            plt.subplot(4, 2, 1)
            plt.plot(data["reward"])
            plt.title("Training Rewards (average)")
            plt.xlabel("Episode/Iteration")
            plt.ylabel("Reward")

            plt.subplot(4, 2, 2)
            plt.plot(data["step_count"])
            plt.title("Max Step Count (training)")
            plt.xlabel("Episode/Iteration")
            plt.ylabel("Steps")

            plt.subplot(4, 2, 3)
            plt.plot(data["eval reward (sum)"])
            plt.title("Return (test)")
            plt.xlabel("Evaluation Run")
            plt.ylabel("Return")

            plt.subplot(4, 2, 4)
            plt.plot(data["eval step_count"])
            plt.title("Max Step Count (test)")
            plt.xlabel("Evaluation Run")
            plt.ylabel("Steps")

            plt.subplot(4, 2, 5)
            plt.plot(data["loss_objective"])
            plt.title("Policy Objective Loss")
            plt.xlabel("Optimization Step/Batch")
            plt.ylabel("Loss Value")

            plt.subplot(4, 2, 6)
            plt.plot(data["loss_critic"])
            plt.title("Critic (Value) Loss")
            plt.xlabel("Optimization Step/Batch")
            plt.ylabel("Loss Value")

            plt.subplot(4, 2, 7)
            plt.plot(data["loss_entropy"])
            plt.title("Entropy Loss")
            plt.xlabel("Optimization Step/Batch")
            plt.ylabel("Loss Value")

            plt.subplot(4, 2, 8)
            plt.plot(data["loss_auxiliary"])
            plt.title("Auxiliary Loss")
            plt.xlabel("Optimization Step/Batch")
            plt.ylabel("Loss Value")

            plt.tight_layout()

            plt.savefig(f"{logging_directory}/plots/PPO_{time.time()}.jpg")
            plt.close('all')

    # simulation.close()
    logger.info('Finished')