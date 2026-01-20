import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def plot_reward(rewards_history: list, window: int, model_name: str):

    plt.figure(figsize=(10, 5))

    # plot the raw rewards
    plt.plot(rewards_history, color="blue", alpha=0.25, label="Episode reward")

    # plot the smoothed rewards
    smoothed = moving_average(rewards_history, window)
    plt.plot(np.arange(window-1, len(rewards_history)), smoothed, color='blue', linewidth=1, label=f"{model_name} average reward")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training reward with env: reaching task of robot with 6 DOF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()