import torch
import torch.nn as nn
from torch.distributions import Normal

class Memory:
    def __init__(self):
        self.states   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []
        self.dones    = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: list[int], action_dim: int):
        super().__init__()
        
        # Actor net
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]), 
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.Tanh(),
            nn.Linear(hidden_dim[2], action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic net
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.Tanh(),
            nn.Linear(hidden_dim[2], 1)
        )
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory: Memory, device):
        state = torch.from_numpy(state).float().to(device)

        mean = self.actor_mean(state)
        std = torch.exp(self.log_std)

        dist = Normal(mean, std)
        action = dist.sample()

        # Tanh squash (important for robot control)
        action_tanh = torch.tanh(action)

        # Log prob correction for tanh
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        memory.states.append(state.squeeze(0))
        memory.actions.append(action_tanh.squeeze(0))
        memory.logprobs.append(log_prob.squeeze(0))

        return action_tanh.squeeze(0).detach().cpu().numpy()
    
    def evaluate(self, states, actions):
        mean = self.actor_mean(states)
        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        # Inverse tanh for log-prob
        atanh_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))

        log_probs = dist.log_prob(atanh_actions)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        state_values = self.critic(states).squeeze(-1)

        return log_probs, torch.squeeze(state_values), entropy


class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, betas, gamma, k_epochs, eps_clip, device):
        self.lr       = lr
        self.betas    = betas
        self.gamma    = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device   = device

        self.policy = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def update(self, memory: Memory, gae_lambda=0.95):

        # convert lists to tensors
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        
        # compute state values for all states (used in GAE)
        with torch.no_grad():
            state_values = self.policy.critic(old_states).squeeze(-1)

        # ----------------------------------
        # Compute advantages using GAE
        # ----------------------------------
        rewards = memory.rewards
        dones   = memory.dones
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else state_values[t]
            else:
                next_value = state_values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - state_values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # targets for critic
        returns = advantages + state_values

        # ----------------------------------
        # Optimize policy for k epochs
        # ----------------------------------
        for _ in range(self.k_epochs):
            logprobs, state_values_pred, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # PPO loss: policy + value + entropy
            loss = -torch.min(surr1, surr2) \
                + 0.5 * self.MseLoss(state_values_pred, returns) \
                - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


    # def update(self, memory: Memory):

    #     rewards = []
    #     discounted_reward = 0

    #     for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
    #         if done:
    #             discounted_reward = 0
            
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         rewards.insert(0, discounted_reward)
        
    #     # normalizing the rewards
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    #     # convert list to tensor
    #     old_states = torch.stack(memory.states).to(self.device).detach()
    #     old_actions = torch.stack(memory.actions).to(self.device).detach()
    #     old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

    #     # optimize policy for k epochs
    #     for _ in range(self.k_epochs):

    #         # evaluating old actions and values
    #         logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

    #         # find the ratio of pi(theta) / pi(theta_old)
    #         ratios = torch.exp(logprobs - old_logprobs.detach())

    #         # find the surrogate loss
    #         advantages = rewards - state_values.detach()
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    #         loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01*dist_entropy

    #         # take the gradient step
    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         self.optimizer.step()

        
    #     # copy the new weights into old policy
    #     self.policy_old.load_state_dict(self.policy.state_dict())
    