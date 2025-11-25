import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from agent import Agent


# Hyperparameters
N_STATES = 52
N_ACTIONS = 12

BATCH_SIZE = 64
EPOCHS = 400
LEARNING_RATE = 1e-4

DATA_FILE = "./booster_dataset/collected.npz" #update for different npz files

# Load the data
data = np.load(DATA_FILE)
observations = data["observations"][:, :24]
actions = data["actions"]

# --- ADD DEBUGGING LINES HERE ---
if len(observations) == 0:
    print("FATAL ERROR: The observations array is empty!")
    print("Check if you successfully recorded data using collect_data.py before pressing ESC.")
    exit()

# Convert to PyTorch Tensors (Pad the remaining components with zeros since they are not needed right now)
X = torch.tensor(
    np.tanh(np.hstack([observations, np.zeros((observations.shape[0], N_STATES - observations.shape[1]))])),
    dtype=torch.float32
)
Y = torch.tensor(actions, dtype=torch.float32)

# Split data for validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print(f"Total Samples: {len(X)}")
print(f"Observation Shape: {X_train.shape} (Input)")
print(f"Action Target Shape: {Y_train.shape} (Output)")

model = Agent(N_STATES, N_ACTIONS)

# Setup loss and optimizer
criterion = nn.MSELoss() # Mean Squared Error is standard for regression tasks like this
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create DataLoader for efficient batching
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
print("Starting Imitation Learning Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_Y in train_loader:

        # Forward pass
        predicted_actions = model(batch_X)[0]
        loss = criterion(predicted_actions, batch_Y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# --- Save Weights ---
SAVE_PATH = "./data/il_actor_seed_weights.pt"
# Save ONLY the state dictionary of the policy network
torch.save(model.state_dict(), SAVE_PATH)
print(f"Imitation Policy weights saved to {SAVE_PATH}")