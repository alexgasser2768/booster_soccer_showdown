import yaml, time, logging
import numpy as np

from src.utils.teleop import teleop
from src.utils.visualize_model import visualize
from src.learning.behavior_cloning import behavior_cloning

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_directory = config['dataset_directory']
    weights_directory = config['weights_directory']
    logging_directory = config['logging_directory']

    logging.basicConfig(filename=f"{logging_directory}/{time.time()}.log", level=logging.INFO)
    logger.info('Started')

    n_states = config['model']['states']
    n_actions = config['model']['actions']

    if config['teleop']['enabled']:
        file_prefix = config['teleop']['file_prefix']
        pos_sensitivity = config['teleop']['position_sensitivity']
        rot_sensitivity = config['teleop']['rotation_sensitivity']

        dataset = teleop(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

        # Save the collected dataset
        dataset_path = f"{dataset_directory}/{file_prefix}-{time.time()}.npz"
        np.savez_compressed(
            dataset_path,
            observations=np.array(dataset["observations"]),
            infos=np.array(dataset["infos"]),
            actions=np.array(dataset["actions"])
        )

        print(f"Dataset saved to {dataset_path}")

    if config["visualize"]["enabled"]:
        weight_path = f"{weights_directory}/{config['visualize']['weight_file']}"

        visualize(
            weight_path=weight_path,
            n_states=n_states,
            n_actions=n_actions
        )

    if config['behavior_cloning']['enabled']:
        data_file = config['behavior_cloning']['data_file']
        batch_size = config['behavior_cloning']['batch_size']
        epochs = config['behavior_cloning']['epochs']
        learning_rate = config['behavior_cloning']['learning_rate']
        file_prefix = config['behavior_cloning']['file_prefix']

        behavior_cloning(
            data_file=f"{dataset_directory}/{data_file}",
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            n_states=n_states,
            n_actions=n_actions,
            model_weights_directory=weights_directory,
            model_weights_prefix=file_prefix
        )

    logger.info('Finished')