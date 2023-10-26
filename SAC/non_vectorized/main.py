import torch
from torch.optim import AdamW
import numpy as np
import gymnasium as gym
from itertools import chain
from models import Q_Network, Policy_Network
from utils import ReplayBuffer
from config import device
from copy import deepcopy
from tqdm import tqdm
import sys


def disable_grads(neural_net):
    for parameter in neural_net.parameters():
        parameter.requires_grad=False
    return neural_net

def enable_grads(neural_net):
    for parameter in neural_net.parameters():
        parameter.requires_grad=True
    return neural_net

def collect_data(p_net, steps):
    with torch.no_grad():
        s, _ = env.reset()
        for _ in range(steps):
            a = p_net.act(torch.as_tensor(s).to(device)).cpu().numpy()
            ns, r, te, tr, _ = env.step(a)
            d = te or tr
            replay_buffer.push(s, a, r, ns, d)
            if d:
                s, _ = env.reset()
            else:
                s = ns

def update():
    for i in range(update_every):
        data = replay_buffer.sample(batch_size)
        s, a, r, ns, t = data["s"], data["a"], data["r"], data["ns"], data["t"]

        with torch.no_grad():
            target_actions = p_net.act(ns)
            log_probs = p_net.get_log_probs(ns, target_actions)
            values = [q_target(ns, target_actions) for q_target in q_targets]
            min_value = torch.minimum(*values)
            target = r + gamma*(1-t)*(min_value - alpha*log_probs)

        
        for j in range(n_q_nets):
            q_optimizers[j].zero_grad()
            q_loss = ((q_nets[j](s, a)-target)**2).mean()
            q_loss.backward()
            q_optimizers[j].step()
              
        for q_net in q_nets:
            disable_grads(q_net)

        action = p_net.act(ns, rsample=True)
        for j in range(n_q_nets):
            values = [q_net(ns, action) for q_net in q_nets]

        for q_net in q_nets:
            enable_grads(q_net)

        min_value = torch.minimum(*values)
        p_optimizer.zero_grad()
        loss = -(min_value - alpha*p_net.get_log_probs(ns, action)).mean()
        loss.backward()
        p_optimizer.step()
                
        with torch.no_grad():
            for k in range(n_q_nets):
                for p, t in zip(q_nets[k].parameters(),
                                q_targets[k].parameters()):
                    t.mul_(polyak)
                    t.add_((1-polyak)*p)


env = gym.make("Pendulum-v1")
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]


gamma = 0.99
buffer_size = 1000
alpha = 0.2
n_q_nets = 2
hidden_size = 128
epochs = 50
steps_per_epoch = 5000
update_every = 100
batch_size = 128
polyak = 0.995

q_nets = [Q_Network(state_dim, hidden_size, action_dim).to(device)
          for _ in range(n_q_nets)]
q_targets = [disable_grads(deepcopy(net)).to(device) for net in q_nets]
q_optimizers = [AdamW(net.parameters(), lr=1e-3) for net in q_nets]
p_net = Policy_Network(state_dim, hidden_size, action_dim).to(device)
p_optimizer = AdamW(p_net.parameters(), lr=1e-3)

replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

collect_data(p_net, buffer_size)

s, _ = env.reset()
rewards_per_epoch = []
durations = []
reward_per_episode = 0
episode_duration = 0
for t in tqdm(range(1, epochs*steps_per_epoch+1)):
    a = p_net.act(torch.as_tensor(s).to(device)).cpu().numpy()
    ns, r, te, tr, _ = env.step(a)
    d = te or tr
    replay_buffer.push(s, a, r, ns, d)
    reward_per_episode += r
    episode_duration +=1
    if d:
        s, _ = env.reset()
        rewards_per_epoch.append(reward_per_episode)
        durations.append(episode_duration)
        reward_per_episode = 0
        episode_duration = 0
    else:
        s = ns

    if t % update_every == 0:
        update()

    if t % steps_per_epoch == 0: 
        average_reward = sum(rewards_per_epoch)/sum(durations)
        print(f"Epoch: {t//steps_per_epoch}\tReward: {average_reward:.3}")
        test_env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
        with torch.no_grad():
            test_state, _ = test_env.reset()
            done = False
            while not done:
                test_action = p_net.act(torch.as_tensor(test_state).to(device)).cpu().numpy()
                test_state, _, d1, d2, _ = test_env.step(test_action)
                done = d1 or d2
        test_env.close()
