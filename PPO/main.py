import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from utils import SetOfTrajectories
from model import Policy_Network, Value_Network


def update():
    s, a, g, A = data["s"], data["a"], data["g"], data["A"]
    with torch.no_grad():
        old_log_probs = policy_net.get_log_probs(s, a)
   
    for _ in range(30):
        new_log_probs = policy_net.get_log_probs(s, a)
        ratio = torch.exp(new_log_probs-old_log_probs)
        uncliped = ratio*A
        cliped = torch.clip(ratio, 0.8, 1.2)*A
        policy_optimizer.zero_grad()
        loss = -(torch.minimum(uncliped, cliped)).mean()
        loss.backward()
        policy_optimizer.step()

    for _ in range(80):
        value_optimizer.zero_grad()
        loss = ((value_net(s)-g)**2).mean()
        loss.backward()
        value_optimizer.step()



epochs = 100
steps_per_epoch = 4000
SET = SetOfTrajectories(3, 1, 4000)
policy_net = Policy_Network(3, 32, 1)
value_net = Value_Network(3, 32)
policy_optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-3)
value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=1e-3)
env = gym.make("Pendulum-v1")
state, _ = env.reset()
for epoch in range(epochs):
    rewards = 0
    n_episodes = 0
    state, _ = env.reset()
    for t in range(steps_per_epoch):
        action = policy_net.act(torch.tensor(state)).numpy()
        next_state, reward, terminated, truncated, info = env.step(action)
        value = value_net.evaluate(torch.tensor(state)).numpy()
        SET.push(state, action, reward, value)
        rewards+=reward.item()
        if truncated or terminated:
            n_episodes+=1
            if truncated:
                value = value_net.evaluate(torch.tensor(next_state)).numpy()
                SET.finish_trajectory(value)
                state, _ = env.reset()
            elif terminated:
                SET.finish_trajectory(0.0)
                state, _ = env.reset()
            elif t == steps_per_epoch-1:
                value = value_net.evaluate(torch.tensor(next_state)).numpy()
                SET.finish_trajectory(value)
                break
        state = next_state
    data = SET.get() 
    update()
    print(f"Epoch : {epoch+1}\tReward : {(rewards/n_episodes):.3}")
    if epoch % 20 == 0:
        with torch.no_grad():
            test_env = gym.make("Pendulum-v1", render_mode="human")
            test_state, _ = test_env.reset()
            done = False
            while not done:
                test_action = policy_net.act(torch.tensor(test_state)).numpy()
                test_state, _, terminated, truncated, _ = test_env.step(test_action)
                done = terminated or truncated
            test_env.close()


            
