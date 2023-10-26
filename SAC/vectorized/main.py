import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
from itertools import chain
from typing import Dict
from random import sample


device=torch.device("cpu")

class Replay_Buffer:
    def __init__(
            self,
            max_size:int,
            batch_size:int,
            n_envs:int
            ) -> None:

        self.transitions = namedtuple("transitions", "s, a, r, ns, t")
        self.memory = deque([], maxlen=max_size)
        self.batch_size = batch_size
        self.n_envs = n_envs

    def push(
            self,
            state:np.ndarray,
            action:np.ndarray,
            reward:np.ndarray,
            next_state:np.ndarray,
            terminated:np.ndarray
            ) -> None:
       self.memory.append(
               self.transitions(
                   state,
                   action,
                   reward,
                   next_state,
                   terminated
                   )
               ) 

    def sample(
            self,
            ) -> Dict[str, torch.Tensor]:
        batch = sample(self.memory, batch_size)
        data = self.transitions(*zip(*batch))
        data =  {k:torch.as_tensor(np.array(v)).view(
            self.batch_size*self.n_envs, -1
            ).to(device)
                for k, v in data._asdict().items()}
        data["t"] = data["t"].type(torch.int32)
        return data

class Actor(
        nn.Module
        ):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            n_hidden_layers,
            lr=1e-3,
            device="cpu"
            ):
        super().__init__()
        self.device = torch.device(device)
        hidden_layers = [nn.Linear(hidden_dim, hidden_dim),
                         nn.ReLU()] * n_hidden_layers
        actor_layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()] +\
                hidden_layers + [nn.Tanh()]
        self.MLP = nn.Sequential(*actor_layers)
        self.mean_layer = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.std_layer = nn.Linear(hidden_dim, action_dim).to(self.device)

        self.optimizer = AdamW(self.parameters(), lr=lr)

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        shared_output = self.MLP(torch.as_tensor(state))
        return (
                self.mean_layer(shared_output),
                self.std_layer(shared_output)
                )

    def act(self, state, rsample=False) -> torch.Tensor:
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


class Critic(
        nn.Module
        ):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            n_hidden_layers,
            lr=1e-3,
            device="cpu"
            ):
        super().__init__()
        self.device = torch.device(device)
        hidden_layers = [nn.Linear(hidden_dim, hidden_dim),
                         nn.ReLU()] * n_hidden_layers
        critic_layers = [nn.Linear(state_dim+action_dim,
                                   hidden_dim),
                         nn.ReLU()]
        critic_layers += hidden_layers + [nn.Linear(hidden_dim, 1)]

        self.q1 = nn.Sequential(*critic_layers).to(self.device)
        self.q2 = nn.Sequential(*critic_layers).to(self.device)

        self.optimizer = AdamW(self.parameters(), lr=lr)

    def forward(self, state, action) -> tuple[torch.Tensor, torch.Tensor]:
        return (
                self.q1(torch.cat([state, action], dim=-1)),
                self.q2(torch.cat([state, action], dim=-1))
                )

    def disable_grad(self) -> None:
        q_params = chain(
                self.q1.parameters(),
                self.q2.parameters()
                )
        for param in q_params:
            param.requires_grad = False

    def enable_grad(self) -> None:
        q_params = chain(
                self.q1.parameters(),
                self.q2.parameters()
                )
        for param in q_params:
            param.requires_grad = True


        
def critic_loss(
        data: Dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor]:
    states = data["s"]
    actions = data["a"]
    new_states = data["ns"]
    r = data["r"]
    t = data["t"]
    with torch.no_grad():
        min_value = torch.min(*critic(states, actions))
        on_policy_actions = actor.act(new_states)
        log_probs = actor.get_log_probs(new_states, on_policy_actions)
        target = r + gamma*(1-t)*(min_value - alpha*log_probs)
    q1_value, q2_value = critic(states, actions)
    return (
            ((q1_value - target)**2).mean(),
            ((q2_value - target)**2).mean()
            )

def actor_loss(
        data: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
    states = data["s"]
    critic.disable_grad()
    on_policy_actions = actor.act(states, rsample=True)
    min_value = torch.min(*critic(states, on_policy_actions))
    log_probs = actor.get_log_probs(states, on_policy_actions)
    critic.enable_grad()
    return -(min_value - alpha*log_probs).mean()


num_envs = 8
batch_size = 4
buffer_size = 100
gamma = 0.99
alpha = 0.2
buffer = Replay_Buffer(buffer_size, batch_size, num_envs)
envs = gym.make_vec("BipedalWalker-v3", num_envs=num_envs)
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

actor = Actor(state_dim, action_dim, 32, 2)
critic = Critic(state_dim, action_dim, 32, 2)

states, _ = envs.reset()
for _ in range(10):
    actions = envs.action_space.sample()
    ns, r, t, _, _ = envs.step(actions)
    buffer.push(states, actions, r, ns, t)
data = buffer.sample()
print(actor_loss(data))
q1_loss, q2_loss = critic_loss(data)
q1_loss.backward()
print(q1_loss.requires_grad)
