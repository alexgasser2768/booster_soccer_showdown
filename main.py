import numpy as np

data = np.load("booster_dataset/jogging.npz")
pos = data["qpos"][:, :23]
vel = data["qvel"][:, :23]

print(np.min(vel), np.median(vel), np.mean(vel), np.max(vel), np.std(vel))
print(np.quantile(vel, 0.01), np.quantile(vel, 0.25), np.quantile(vel, 0.5), np.quantile(vel, 0.75), np.quantile(vel, 0.99))
print()

vel = np.tanh(vel)
print(np.min(vel), np.median(vel), np.mean(vel), np.max(vel), np.std(vel))
print(np.quantile(vel, 0.01), np.quantile(vel, 0.25), np.quantile(vel, 0.5), np.quantile(vel, 0.75), np.quantile(vel, 0.99))