import logging
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .agent import Agent
from ..booster_control import create_input_vector

logger = logging.getLogger(__name__)


def behavior_cloning(data_files: str, batch_size: int, epochs: int, learning_rate: float, n_states: int, n_actions: int, model_weights_directory: str, model_weights_prefix: str):
    observations = None
    infos = None
    actions = None

    # Load the data
    for d_file in data_files:
        data = np.load(d_file, allow_pickle=True)
        observations = data["observations"][:, :24] if observations is None else np.concatenate([observations, data["observations"][:, :24]])
        infos = data["infos"] if infos is None else np.concatenate([infos, data["infos"]])
        actions = data["actions"] if actions is None else np.concatenate([actions, data["actions"]])

    if len(observations) == 0 or len(actions) == 0:
        logger.error("The observations array is empty.")
        return

    X = torch.tensor(
        np.array([create_input_vector(infos[i], observations[i]) for i in range(len(observations))]),
        dtype=torch.float32
    )
    Y = torch.tensor(actions, dtype=torch.float32)

    # Split data for validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    logger.info(f"Total Samples: {len(X)}")
    logger.info(f"Observation Shape: {X_train.shape} (Input)")
    logger.info(f"Action Target Shape: {Y_train.shape} (Output)")

    model = Agent(n_states, n_actions)

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
            total_loss = 0

            for batch_X, batch_Y in train_loader:
                # Forward pass
                predicted_actions = model(batch_X)
                loss = criterion(predicted_actions, batch_Y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute loss
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            test_predicted = model(X_test)
            test_loss = criterion(test_predicted, Y_test).item()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}")
    except KeyboardInterrupt:
        pass

    # --- Save Weights ---
    weight_filename = model.saveWeights(model_weights_directory, prefix=model_weights_prefix)

    logger.info(f"Model weights saved to {weight_filename}")