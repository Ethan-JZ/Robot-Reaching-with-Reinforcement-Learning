# The agent_ddpg.py consists of 6 components
#
#        ##############      #####################
#        # Critic Net #      # Target Critic Net #
#        ##############      #####################
#
#        #############       ####################
#        # Actor Net #       # Target Actor Net #
#        #############       ####################
#
#        #################   ##############
#        # Replay Buffer #   # DDPG agent #
#        #################   ##############

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import torch.nn.functional as F
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: list, action_dim: int):
        super().__init__()

        # layer structure: state - hidden - hidden - action
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])  # state - hidden 
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1]) # hidden - hidden 
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2]) # hidden - hidden 
        self.fc4 = nn.Linear(hidden_dim[2], action_dim) # hidden - action
    
    def forward(self, x):
        """
        forward propagation based on input state x, then return action x
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) # tanh controls output to [-1, 1]

        return x


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: list, action_dim: int):
        super().__init__()

        # layer structure: (state + action) - hidden - hidden - Q value (1 dim)
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1]) # hidden - hidden 
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2]) # hidden - hidden 
        self.fc4 = nn.Linear(hidden_dim[2], 1)

    
    def forward(self, x, action):
        x = torch.cat([x, action], 1)  # concatenate on feature dimension, row: sample dimension, col: feature dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """
    Add the experience to this class
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add_memo(self, state, action, reward, state_next, done):
        """
        Add experience to the buffer
        """
        state      = np.expand_dims(state, 0)
        state_next = np.expand_dims(state_next, 0)

        self.buffer.append((state, action, reward, state_next, done))
    
    def sample(self, batch_size: int):
        """
        Sample a mini batch of experiences from the buffer
        """
        state, action, reward, state_next, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(state_next), done
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, 
                 state_dim: int, 
                 hidden_dim: list[int], 
                 action_dim: int, 
                 actor_lr: float, 
                 critic_lr: float, 
                 gamma: float, 
                 device, 
                 memory_size: int,
                 batch_size: int,
                 tau: float):
        """
        construct the ddpg agent with hyper parameters
        state_dim: layer state dimension number
        hidden_dim: layer hidden dimension [n, n, n]
        action_dim: layer action dimension
        actor_lr: learning rate of the actor net
        critic_lr: learning rate of the critic net
        gamma: discounted return factor
        device: device to do the computation
        memory_size: size of the replay buffer
        batch_size: mini batch size
        tau: tau to update parameters
        """

        self.device      = device
        self.gamma       = gamma
        self.memory_size = memory_size
        self.batch_size  = batch_size
        self.tau         = tau
        
        # init actor part
        self.actor           = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target    = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # init critic part
        self.critic           = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target    = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # init a replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
    
    def get_action(self, state):
        """
        choose action based on state
        """

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # change the dim from (n, ) to (1, n)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]  # this will return (action_dim, ) which is expected for the environment
    
    def update(self):
        
        # if not reaching the batch size, do nothing
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # if reaching the batch size, sample batch size of experiences
        states, actions, rewards, states_next, dones = self.replay_buffer.sample(self.batch_size)

        # change to tensors and move to GPU for computation
        states      = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions     = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards     = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        states_next = torch.from_numpy(np.array(states_next, dtype=np.float32)).to(self.device)
        dones       = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        # Update critic by computing the loss and minimize the loss
        actions_next = self.actor_target(states_next)
        target_Q = self.critic_target(states_next, actions_next.detach())
        target_Q = rewards + self.gamma * target_Q * (1 - dones)
        current_Q = self.critic(states, actions)
        
        ## compute loss and minimize the loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()  # clear old gradients
        critic_loss.backward()  # compute the loss derivates
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # compute the loss derivates
        self.actor_optimizer.step()

        # Update the target networks of critic and actor
        for target_param, parm in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * parm.data + (1 - self.tau) * target_param.data)

        for target_param, parm in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * parm.data + (1 - self.tau) * target_param.data)

