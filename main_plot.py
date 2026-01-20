import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self, 
                 td3_data_path: str, 
                 ddpg_data_path: str,
                 ppo_data_path: str):
        self.td3_data  = self._read_data(td3_data_path)
        self.ddpg_data = self._read_data(ddpg_data_path)
        self.ppo_data  = self._read_data(ppo_data_path)
        self.window = 50

        # plot the rewards
        self._plot_rewards()
    
    def _read_data(self, npy_data: np.ndarray) -> np.ndarray:
        data = np.load(npy_data)
        return data

    def _moving_average(self, x):
        return np.convolve(x, np.ones(self.window)/self.window, mode="valid")

    def _plot_rewards(self):

        plt.figure(figsize=(10, 5))

        # plot the raw rewards
        plt.plot(self.ddpg_data, color="blue", alpha=0.25, label="DDPG episode rewards")
        plt.plot(self.td3_data, color="red", alpha=0.25, label="TD3 episode rewards")
        plt.plot(self.ppo_data, color="green", alpha=0.25, label="PPO episode rewards")
        

        # plot the smoothed rewards
        ddpg_smoothed = self._moving_average(self.ddpg_data)
        td3_smoothed  = self._moving_average(self.td3_data)
        ppo_smoothed  = self._moving_average(self.ppo_data)
        
        plt.plot(np.arange(self.window-1, len(self.ddpg_data)), ddpg_smoothed, color='blue', linewidth=1, label=f"DDPG episode average reward")
        plt.plot(np.arange(self.window-1, len(self.td3_data)), td3_smoothed, color='red', linewidth=1, label=f"TD3 episode average reward")
        plt.plot(np.arange(self.window-1, len(self.ppo_data)), ppo_smoothed, color='green', linewidth=1, label=f"PPO episode average reward")
        

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training reward with env: reaching task of robot with 6 DOF")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ddpg_npy_str = "episode_rewards_ddpg.npy"
    td3_npy_str  = "episode_rewards_td3.npy"
    ppo_npy_str  = "episode_rewards_ppo.npy"
    plot_obj = Plotting(td3_data_path=td3_npy_str, ddpg_data_path=ddpg_npy_str, ppo_data_path=ppo_npy_str)