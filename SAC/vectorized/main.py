import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
from itertools import chain
from typing import Dict, Generator
from random import sample
from copy import deepcopy
from tqdm import tqdm
import sys



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
        batch = sample(self.memory, self.batch_size)
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
                hidden_layers[:-1] + [nn.Tanh()]
        self.MLP = nn.Sequential(*actor_layers)
        self.mean_layer = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.std_layer = nn.Linear(hidden_dim, action_dim).to(self.device)

        #self.optimizer = AdamW(self.parameters(), lr=lr)

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
        #self.q2 = nn.Sequential(*critic_layers).to(self.device)
        self.q2 = deepcopy(self.q1)

        self.optimizer_q1 = AdamW(self.q1.parameters(), lr=lr)
        self.optimizer_q2 = AdamW(self.q2.parameters(), lr=lr)

    def forward(self, state, action) -> tuple[torch.Tensor]:
        return (
                self.q1(torch.cat([state, action], dim=-1)),
                self.q2(torch.cat([state, action], dim=-1))
                )

    def disable_grad(self) -> None:
        params = self.get_params()
        for param in params:
            param.requires_grad = False

    def enable_grad(self) -> None:
        params = self.get_params()
        for param in params:
            param.requires_grad = True

    def get_params(self) -> Generator[
            torch.nn.parameter.Parameter,
            None,
            None
            ]:
        return chain(
                self.q1.parameters(),
                self.q2.parameters()
                )


        
def critic_loss(
        data: Dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
    states = data["s"]
    actions = data["a"]
    next_states = data["ns"]
    r = data["r"]
    t = data["t"]
    with torch.no_grad():
        on_policy_actions = actor.act(next_states)
        log_probs = actor.get_log_probs(next_states, on_policy_actions)
        min_value = torch.minimum(
                *critic_target(next_states, on_policy_actions)
                )
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
    on_policy_actions = actor.act(states, rsample=True)
    critic.disable_grad()
    min_value = torch.minimum(
            *critic(states, on_policy_actions),
            )
    critic.enable_grad()

    log_probs = actor.get_log_probs(states, on_policy_actions)
    return -(min_value - alpha*log_probs).mean()

def update(data):
    q1_loss, q2_loss = critic_loss(data)

    critic.optimizer_q1.zero_grad()
    q1_loss.backward()
    critic.optimizer_q1.step()

    critic.optimizer_q2.zero_grad()
    q2_loss.backward()
    critic.optimizer_q2.step()

    policy_loss = actor_loss(data)
    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()

    non_target = critic.get_params()
    target = critic_target.get_params()

    with torch.no_grad():
        for p, t in zip(non_target, target):
            t.mul_(polyak)
            t.add_((1-polyak)*p) 


def test_agent(
        env_name:str,
        n_test_steps:int,
        render_mode=None) -> float:
    total_reward = 0
    total_episodes = 0
    test_env = gym.make_vec(env_name, num_envs=1)
    for _ in range(n_test_steps):
        state, _ = test_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = actor.act(torch.as_tensor(state)).numpy()
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                total_episodes += 1
    test_env.close()
    if render_mode == "human":
        test_env = gym.make_vec(env_name, num_envs=1, render_mode="human")
        state, _ = test_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = actor.act(torch.as_tensor(state)).numpy()
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
        test_env.close()
    return total_reward / total_episodes

def collect_data(collect_steps:int) -> None:
    states, _ = envs.reset()
    for t in range(collect_steps):
        with torch.no_grad():
            actions = actor.act(states).cpu().numpy()
        next_states, rewards, terminated, truncated, _ = envs.step(actions)
        buffer.push(states, actions, rewards, next_states, terminated) 
        states = next_states


epochs = 50
steps_per_epoch = 1000
num_envs = 8
batch_size = 8
buffer_size = 10000
gamma = 0.99
alpha = 0.2
polyak = 0.995
update_every = 50
collect_steps = 1000
buffer = Replay_Buffer(buffer_size, batch_size, num_envs)
envs = gym.make_vec("Pendulum-v1", num_envs=num_envs)
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

actor = Actor(state_dim, action_dim, 64, 2)
actor_optimizer = AdamW(actor.parameters(), lr=1e-3)

critic = Critic(state_dim, action_dim, 64, 2)
critic_target = deepcopy(critic)
critic_target.disable_grad()

collect_data(collect_steps)
for epoch in range(epochs):
    states, _ = envs.reset()
    for t in tqdm(range(steps_per_epoch)):
        with torch.no_grad():
            actions = actor.act(states).cpu().numpy()
        next_states, rewards, terminated, truncated, _ = envs.step(actions)
        buffer.push(states, actions, rewards, next_states, terminated) 
        states = next_states
        if t % update_every == 0:
            for j in range(update_every):
                data = buffer.sample()
                update(data)
        
    average_reward = test_agent("Pendulum-v1", 10)
    print(average_reward)
