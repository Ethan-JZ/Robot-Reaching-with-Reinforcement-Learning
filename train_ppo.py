import numpy as np
import torch
import os
import random

from env.robot_reaching_pos_dense_rewards import RobotEnv
from agents.agent_ppo import PPOAgent, Memory
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
    NUM_EPISODES = 5000
    NUM_STEPS    = env.max_steps
    
    GAMMA        = 0.99
    LR           = 3e-4
    EPS_CLIP     = 0.2
    K_EPOCHS     = 10
    BETAS        = (0.9, 0.999)
    
    UPDATE_TIME  = 4000
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
    memory = Memory()
    agent = PPOAgent(state_dim=STATE_DIM, 
                     hidden_dim=HIDDEN_DIM, 
                     action_dim=ACTION_DIM, 
                     lr=LR, 
                     betas=BETAS, 
                     gamma=GAMMA, 
                     k_epochs=K_EPOCHS, 
                     eps_clip=EPS_CLIP, 
                     device=DEVICE)
    
    reward_history = np.empty(shape=NUM_EPISODES)  # initialize a list to save reward for each episode
    
    # --------------------------
    # Training loop
    # --------------------------
    time_step = 0
    for episode_i in range(NUM_EPISODES):
       
        # ---------------------------------
        # Reset the env and get the states
        # ---------------------------------
        obs_dict, _    = env.reset()
        state          = build_state(obs_dict) 
        episode_reward = 0
        
        for step_i in range(NUM_STEPS):

            time_step += 1
            
            # --------------------------
            # Running policy old
            # --------------------------
            action = agent.policy_old.act(state, memory, DEVICE)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # -----------------------------
            # Step the environment
            # -----------------------------
            obs_dict_next, reward, terminated, truncated, info = env.step(action)
            state_next = build_state(obs_dict_next)
            
            # -----------------------------
            # Store experience
            # -----------------------------
            memory.rewards.append(reward)
            memory.dones.append(terminated)

            # -----------------------------
            # Update agent it it's time
            # -----------------------------
            if time_step % UPDATE_TIME == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0

            # -----------------------------
            # Move to the next state
            # -----------------------------
            state          = state_next
            episode_reward += reward
            
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
    torch.save(agent.policy.actor_mean.state_dict(), model + f"ppo_actor_reach.pth")
    np.save("episode_rewards_ppo.npy", np.array(reward_history))

    env.close()

    # plot the training reward history
    plot_reward(reward_history, window=20, model_name="PPO")


if __name__ == "__main__":
    main_train()
