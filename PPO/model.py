import torch
import torch.nn as nn
from torch.distributions import Normal

class Policy_Network(nn.Module):
    def __init__(self, observation_dim, hidden_dim, action_dim):
        super().__init__()
        self.shared_net = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                )
        self.mu_net = nn.Sequential(
                nn.Linear(hidden_dim, action_dim)
                )
        self.std_net = nn.Sequential(
                nn.Linear(hidden_dim, action_dim)
                )
    def forward(self, state):
        shared_outs = self.shared_net(state)
        actions_means = self.mu_net(shared_outs)
        actions_stds = self.std_net(shared_outs)
        return actions_means, actions_stds

    def act(self, state):
        with torch.no_grad():
            means, stds = self.forward(state)
            distribution = Normal(means, torch.log(torch.exp(stds)+1))
            return distribution.sample()

    def get_log_probs(self, states, actions):
        means, stds = self.forward(states)
        distribution = Normal(means, torch.log(torch.exp(stds)+1))
        return distribution.log_prob(actions).squeeze()


class Value_Network(nn.Module):
    def __init__(self, observation_dim, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                )

    def forward(self, state):
        return self.MLP(state).squeeze()

    def evaluate(self, state):
        with torch.no_grad():
            return self.forward(state)


