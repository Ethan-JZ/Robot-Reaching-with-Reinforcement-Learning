import numpy as np
import torch
import os
import random

from stable_baselines3 .common.noise import NormalActionNoise

from env.robot_reaching_pos_dense_rewards import RobotEnv
from agents.agent_ddpg import DDPGAgent
from helpers.plot_rewards import plot_reward


def build_state(obs_dict):
    return np.concatenate([
        obs_dict["observation"],
        obs_dict["achieved_goal"],
        obs_dict["desired_goal"]
    ], axis=0)


def main_train():
    
    # ---------------------
    # Initialize env
    # ---------------------
    xml_path     = "env/robot.xml"
    ws_path      = "env/robot_workspace.json"
    joints_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_left_finger_joint", "gripper_right_finger_joint"]
    env          = RobotEnv(xml_path=xml_path, ws_path=ws_path, joints_names=joints_names, ee_site="ee_site", render=True)

    # -------------------
    # Hyper parameters
    # -------------------
    SEED         = 42
    NUM_EPISODES = 6000
    NUM_STEPS    = env.max_steps
    
    GAMMA        = 0.99
    ACTOR_LR     = 3e-4
    CRITIC_LR    = 3e-4
    TAU          = 0.005
    
    BATCH_SIZE   = 64
    MEMORY_SIZE  = 100000
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    STATE_DIM    = int(env.observation_space['achieved_goal'].shape[0]) + int(env.observation_space['desired_goal'].shape[0]) + int(env.observation_space['observation'].shape[0])
    ACTION_DIM   = int(env.action_space.shape[0])
    HIDDEN_DIM   = [512, 512, 256]

    print("========================================")
    print(f"Device type: {DEVICE}")
    print(f"Action low: {env.action_space.low}")
    print(f"Action high: {env.action_space.high}")
    print("========================================")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # --------------------------------------------
    # Initialize agent and reward history
    # --------------------------------------------
    agent = DDPGAgent(state_dim=STATE_DIM,
                      hidden_dim=HIDDEN_DIM, 
                      action_dim=ACTION_DIM, 
                      actor_lr=ACTOR_LR, 
                      critic_lr=CRITIC_LR, 
                      gamma=GAMMA, 
                      device=DEVICE, 
                      memory_size=MEMORY_SIZE, 
                      batch_size=BATCH_SIZE, 
                      tau=TAU)  
    
    reward_history = np.empty(shape=NUM_EPISODES)  # initialize a list to save reward for each episode
    
    # --------------------------
    # Training loop
    # --------------------------
    for episode_i in range(NUM_EPISODES):
       
        # ---------------------------------
        # Reset the env and get the states
        # ---------------------------------
        obs_dict, _    = env.reset()
        state          = build_state(obs_dict) 
        episode_reward = 0
        
        # --------------------------------------------
        # Noise decay per episode
        # --------------------------------------------
        noise_sigma = max(0.02, 0.1 * (1 - episode_i / NUM_EPISODES))
        noise       = NormalActionNoise(mean=np.zeros(ACTION_DIM), sigma=noise_sigma * np.ones(ACTION_DIM))

        for step_i in range(NUM_STEPS):
            
            # --------------------------
            # Setting action with noise
            # --------------------------
            action = agent.get_action(state) + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # -----------------------------
            # Step environment
            # -----------------------------
            obs_dict_next, reward, terminated, truncated, info = env.step(action)
            state_next = build_state(obs_dict_next)
            
            # -----------------------------
            # Store experience
            # -----------------------------
            agent.replay_buffer.add_memo(state, action, reward, state_next, terminated)

            # -----------------------------
            # Move to the next state
            # -----------------------------
            state          = state_next
            episode_reward += reward
            
            # -----------------------------
            # Update agent
            # -----------------------------
            agent.update()

            if terminated or truncated:
                break
        
        # -----------------------------
        # Save the reward history
        # -----------------------------
        reward_history[episode_i] = episode_reward
        print(f"Episode: {episode_i + 1}, Reward: {episode_reward}")

    # set a path to save models parameters
    current_path = os.path.dirname(os.path.realpath(__file__))
    model        = current_path + '/reach_env_model/'

    # make sure the folder exists
    os.makedirs(model, exist_ok=True)

    # save models
    torch.save(agent.actor.state_dict(), model + f"ddpg_actor_reach.pth")
    torch.save(agent.critic.state_dict(), model + f"ddpg_critic_reach.pth")
    np.save("episode_rewards_ddpg.npy", np.array(reward_history))

    env.close()

    # plot the training reward history
    plot_reward(reward_history, window=20, model_name="DDPG")


if __name__ == "__main__":
    main_train()
