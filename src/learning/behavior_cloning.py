import logging
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .agent import Agent
from ..booster_control import DEVICE, create_input_vector

logger = logging.getLogger(__name__)


def behavior_cloning(data_files: str, batch_size: int, epochs: int, learning_rate: float, n_states: int, n_actions: int, model_weights_directory: str, model_weights_prefix: str, model_weights_file: str):
    observations = None
    infos = None
    actions = None

    logger.info(f"Device used for training: {DEVICE}")

    # Load the data
    for d_file in data_files:
        data = np.load(d_file, allow_pickle=True)

        observations = data["observations"][:, :24] if observations is None else np.concatenate([observations, data["observations"][:, :24]])
        infos = data["infos"] if infos is None else np.concatenate([infos, data["infos"]])
        actions = data["actions"] if actions is None else np.concatenate([actions, data["actions"]])

    if len(observations) == 0 or len(observations) != len(actions) or len(observations) != len(infos):
        logger.error(f"Empty or shape mismatch: (obs = {len(observations)}) (actions = {len(actions)}) (infos = {len(infos)})")
        return

    model_states = np.array([create_input_vector(infos[i], observations[i]) for i in range(len(observations))])

    X = torch.tensor(
        model_states[:-1],
        dtype=torch.float32,
        device=DEVICE
    )
    Y = torch.tensor(
        np.hstack([actions[:-1], model_states[1:]]),
        dtype=torch.float32,
        device=DEVICE
    )

    # Split data for validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    logger.info(f"Total Samples: {len(X)}")
    logger.info(f"Observation Shape: {X_train.shape} (Input)")
    logger.info(f"Action Target Shape: {Y_train.shape} (Output)")

    model = Agent(n_states, n_actions)
    try:
        model.loadWeights(f"{model_weights_directory}/{model_weights_file}")
    except:
        logger.error(f"Couldn't load the weights {model_weights_file}")

    model.to(DEVICE)

    # Setup loss and optimizer
    criterion = nn.MSELoss() # Mean Squared Error is standard for regression tasks like this
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for efficient batching
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Training Loop ---
    logger.info("Starting Imitation Learning Training...")
    try:
        for epoch in tqdm(range(epochs)):
            total_loss, total_action_loss, total_joint_loss = 0, 0, 0

            for batch_X, batch_Y in train_loader:
                # Forward pass
                predicted_actions, predicted_joints = model(batch_X)

                action_loss = criterion(predicted_actions, batch_Y[:, :12])
                joints_loss = criterion(predicted_joints, batch_Y[:, 12:])

                loss = action_loss + joints_loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute loss
                total_action_loss += action_loss.item()
                total_joint_loss += joints_loss.item()
                total_loss += loss.item()

            avg_action_loss = total_action_loss / len(train_loader)
            avg_joint_loss = total_joint_loss / len(train_loader)
            avg_loss = total_loss / len(train_loader)

            test_predicted = model(X_test)
            test_loss = (
                criterion(test_predicted[0], Y_test[:, :12]) + 
                criterion(test_predicted[1], Y_test[:, 12:])
            ).item()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Action Loss: {avg_action_loss:.6f}, Joint Loss: {avg_joint_loss:.6f}, Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}")
    except KeyboardInterrupt:
        pass

    # --- Save Weights ---
    weight_filename = model.saveWeights(model_weights_directory, prefix=model_weights_prefix)

    logger.info(f"Model weights saved to {weight_filename}")