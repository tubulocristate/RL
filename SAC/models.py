import torch
import torch.nn as nn 
from torch.distributions import Normal

class Q_Network(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.MLP = torch.nn.Sequential(
                torch.nn.Linear(state_dim+action_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1),
                )

    def forward(self, state, action):
        return self.MLP(torch.cat([state, action], dim=-1))
    

class Policy_Network(torch.nn.Module):
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim):
        super().__init__()
        self.shared_net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                )
        self.mean_layer = torch.nn.Linear(hidden_dim, 1)
        self.std_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_output = self.shared_net(state)
        return self.mean_layer(shared_output), self.std_layer(shared_output)

    def act(self, state, rsample=False):
        mean, std = self.forward(state)
        distribution = Normal(mean, torch.log(torch.exp(std)+1))
        if not rsample:
            return distribution.sample()
        else:
            return distribution.rsample()

    def get_log_probs(self, state, action):
        mean, std = self.forward(state)
        distribution = Normal(mean, torch.log(torch.exp(std)+1))
        return distribution.log_prob(action)

