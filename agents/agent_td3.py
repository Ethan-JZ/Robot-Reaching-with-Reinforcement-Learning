import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import copy
from collections import deque
import random


class Actor(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: list[int], action_dim: int):
        super().__init__()
        
        # state - hidden - hidden - hidden - action
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], action_dim)

    def forward(self, state):
        """  Forward propagation of Actor network"""
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))

        return torch.tanh(self.fc4(a))
    

class Critic(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: list[int], action_dim: int):
        super().__init__()
        
        # Q1 architecture
        # state + action - hidden - hidden - hidden - 1

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], 1)

        # Q2 architecture
        # state + action - hidden - hidden - hidden - 1
        self.fc5 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.fc6 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc7 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc8 = nn.Linear(hidden_dim[2], 1)
    
    def forward(self, state, action):
        """  Forward propagation of Critic network"""
        state_action = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(state_action))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        q2 = F.relu(self.fc5(state_action))
        q2 = F.relu(self.fc6(q2))
        q2 = F.relu(self.fc7(q2))
        q2 = self.fc8(q2)

        return q1, q2
    
    def Q1(self, state, action):
        """  Q1 Critic network for update using deterministic policy gradient"""
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(state_action))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        return q1


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def sample(self, batch_size: int):
        state, action, reward, state_next, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(state_next), done
    
    def add_memo(self, state, action, reward, state_next, done):   
        state      = np.expand_dims(state, 0)
        state_next = np.expand_dims(state_next, 0)
        self.buffer.append((state, action, reward, state_next, done))
    
    def __len__(self):
        return len(self.buffer)


class TD3Agent:

    def __init__(
            self,
            state_dim: int,
            hidden_dim: list[int],
            action_dim: int,
            gamma_: float,
            tau:float,
            policy_noise: float,
            noise_clip: float,
            policy_freq: int,
            device,
            actor_lr: float,
            critic_lr: float,
            batch_size: int,
            memory_size: int
        ):
        
        # ----------------------------------------
        # Actor network and Target Actor network
        # ----------------------------------------
        self.actor           = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target    = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ------------------------------------------
        # Critic network and Target Critic network
        # ------------------------------------------
        self.critic           = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target    = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ------------------------------
        # hyper parameters setting
        # ------------------------------
        self.gamma_       = gamma_
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq
        self.device       = device
        self.batch_size   = batch_size
        self.memory_size  = memory_size
        
        # -----------------------------------------
        # Initialize total steps and replay buffer
        # -----------------------------------------
        self.total_steps   = 0
        self.replay_buffer = ReplayBuffer(memory_size)
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self):

        self.total_steps += 1

        # if not reaching the batch size, do nothing
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample replay buffer
        state, action, reward, state_next, done = self.replay_buffer.sample(self.batch_size)
        
        # move state, action, reward, state', and done to device
        state      = torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device)
        action     = torch.from_numpy(np.array(action, dtype=np.float32)).to(self.device)
        reward     = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(1).to(self.device)
        state_next = torch.from_numpy(np.array(state_next, dtype=np.float32)).to(self.device)
        done       = torch.from_numpy(np.array(done, dtype=np.float32)).unsqueeze(1).to(self.device)

        with torch.no_grad():

            # select action based on policy and add clip noise
            noise = (torch.rand_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # noise_mag = noise.abs().mean().item()
            # print(noise_mag)

            # next action
            action_next = (self.actor_target(state_next) + noise).clamp(-1, 1)

            # compute the target Q value
            target_Q1, target_Q2 = self.critic_target(state_next, action_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma_ * target_Q
        
        # get the current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimze the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed the policy updates
        if self.total_steps % self.policy_freq == 0:

            # compute actor losses
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the frozen target models
            for param, target_parm in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_parm.data.copy_(self.tau * param.data + (1 - self.tau) * target_parm.data)
            
            for param, target_parm in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_parm.data.copy_(self.tau * param.data + (1 - self.tau) * target_parm.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")



