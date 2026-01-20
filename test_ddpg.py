import torch
import numpy as np
import time
from env.robot_reaching_pos_dense_rewards import RobotEnv
from agents.agent_ddpg import DDPGAgent


def build_state(obs_dict):
    return np.concatenate([
        obs_dict["observation"],
        obs_dict["achieved_goal"],
        obs_dict["desired_goal"]
    ])


def test_ddpg():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

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
    STATE_DIM    = int(env.observation_space['achieved_goal'].shape[0]) + int(env.observation_space['desired_goal'].shape[0]) + int(env.observation_space['observation'].shape[0])
    ACTION_DIM   = int(env.action_space.shape[0])
    HIDDEN_DIM   = [512, 512, 256]

    # ---------------------
    # Initialize agent 
    # ---------------------
    agent = DDPGAgent(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        actor_lr=3e-4,       # irrelevant for testing
        critic_lr=3e-4,      # irrelevant for testing
        gamma=0.99,
        device=DEVICE,
        memory_size=1,       # dummy
        batch_size=1,
        tau=0.0
    )

    # --------------------------
    # Load trained weights
    # --------------------------
    model_path = "reach_env_model/"
    agent.actor.load_state_dict(torch.load(model_path + "ddpg_actor_reach.pth", map_location=DEVICE))
    agent.actor.eval()

    # --------------------------
    # Training loop
    # --------------------------
    NUM_TEST_EPISODES = 100

    for ep in range(NUM_TEST_EPISODES):
        
        # ---------------------------------
        # Reset the env and get the states
        # ---------------------------------
        obs_dict, _ = env.reset(seed=100 + ep)
        state       = build_state(obs_dict)
        
        # ---------------------------------
        # Initialize done and episode reward
        # ---------------------------------
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = agent.actor(state_tensor).cpu().numpy().flatten()

            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # --------------------------------------------------
            # Update the observation after stepping the action
            # --------------------------------------------------
            obs_dict, reward, done, truncated, info = env.step(action)
            state = build_state(obs_dict)
            episode_reward += reward

            # --- slow down for visualization ---
            time.sleep(0.02)

            if done or truncated:
                time.sleep(2)
                break

        print(f"[TEST] Episode {ep + 1} | Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    test_ddpg()