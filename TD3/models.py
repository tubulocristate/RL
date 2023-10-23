import torch
import torch.nn as nn 

class Q_NET(torch.nn.Module):
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
    

class POLICY_NET(torch.nn.Module):
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 action_low,
                 action_high):
        super().__init__()
        self.MLP = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Sigmoid(),
                )
        self.range = action_high-action_low
        self.shift = action_low

    
    def forward(self, state):
        return self.range*self.MLP(state)+self.shift
