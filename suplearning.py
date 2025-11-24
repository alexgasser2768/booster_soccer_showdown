import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

DATA_FILE = "./booster_dataset/jogging.npz" #update for different npz files

# Load the data
data = np.load(DATA_FILE)
q_pos = data["qpos"][:, :12]
q_vel = data["qvel"][:, :12]

observations = np.hstack([q_pos, q_vel])[:-1, :]
actions = np.tanh(q_vel[1:, :])
# observations = data["observations"]
# actions_target = data["actions"]

# --- ADD DEBUGGING LINES HERE ---
if len(observations) == 0:
    print("FATAL ERROR: The observations array is empty!")
    print("Check if you successfully recorded data using collect_data.py before pressing ESC.")
    exit()

# Convert to PyTorch Tensors
X = torch.tensor(observations, dtype=torch.float32)
Y = torch.tensor(actions, dtype=torch.float32)

# Split data for validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print(f"Total Samples: {len(X)}")
print(f"Observation Shape: {X_train.shape} (Input)")
print(f"Action Target Shape: {Y_train.shape} (Output)")

# these are found from the .npz file collected from collect_data.py
N_STATES = 24
N_ACTIONS = 12

class ActorMLP(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ActorMLP, self).__init__()

        # Match the shared network architecture
        self.shared_net = nn.Sequential(
            nn.Tanh(),
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor_head = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.shared_net(x)
        return self.actor_head(features)

model = ActorMLP(N_STATES, N_ACTIONS)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-4

# Setup loss and optimizer
criterion = nn.MSELoss() # Mean Squared Error is standard for regression tasks like this
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create DataLoader for efficient batching
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- Training Loop ---
print("Starting Imitation Learning Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_Y in train_loader:

        # Forward pass
        predicted_actions = model(batch_X)
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
print(f"✅ Imitation Policy weights saved to {SAVE_PATH}")