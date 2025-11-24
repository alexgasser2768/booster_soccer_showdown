import numpy as np


def print_stats(data_points: np.ndarray, variable_name: str = ""):
    print(f"Statistics for raw {variable_name}:")
    print(f"  Min: {np.min(data_points):.4f}, Median: {np.median(data_points):.4f}, Mean: {np.mean(data_points):.4f}, Max: {np.max(data_points):.4f}, Std Dev: {np.std(data_points):.4f}")
    print(f"  Quantiles - 1%: {np.quantile(data_points, 0.01):.4f}, 25%: {np.quantile(data_points, 0.25):.4f}, 50%: {np.quantile(data_points, 0.5):.4f}, 75%: {np.quantile(data_points, 0.75):.4f}, 99%: {np.quantile(data_points, 0.99):.4f}")
    print("-" * 20)

    data_points = np.tanh(data_points)

    print(f"Statistics for normalized {variable_name} (tanh):")
    print(f"  Min: {np.min(data_points):.4f}, Median: {np.median(data_points):.4f}, Mean: {np.mean(data_points):.4f}, Max: {np.max(data_points):.4f}, Std Dev: {np.std(data_points):.4f}")
    print(f"  Quantiles - 1%: {np.quantile(data_points, 0.01):.4f}, 25%: {np.quantile(data_points, 0.25):.4f}, 50%: {np.quantile(data_points, 0.5):.4f}, 75%: {np.quantile(data_points, 0.75):.4f}, 99%: {np.quantile(data_points, 0.99):.4f}")
    print()


data = np.load("booster_dataset/jogging.npz")
pos = data["qpos"][:, :12]
vel = data["qvel"][:, :12]

print_stats(pos, "positions")
print_stats(vel, "velocities")