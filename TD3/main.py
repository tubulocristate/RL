import torch
from torch.optim import AdamW
import numpy as np
import gymnasium as gym
from models import Q_NET, POLICY_NET, Q_NET_TEST
from utils import ReplayBuffer
from copy import deepcopy
import sys
from tqdm import tqdm

env = gym.make('Pendulum-v1', g=9.81)
def TD3(env, epochs, steps_per_epoch, gamma, polyak, batch_size,
        update_every, noise_clip, action_var, target_var
        ):
    assert isinstance(env.action_space, gym.spaces.Box), "TD3 only for environments with continious action space!"
    
    a_low = env.action_space.low.item()
    a_high = env.action_space.high.item()

    def select_action(policy_net,
                      state,
                      noise_clip=noise_clip,
                      a_low=a_low,
                      a_high=a_high,
                      clip=False):
        with torch.no_grad():
            noise_size = (state.size(0), ) if len(state.size()) != 1 else (1, )
            if clip:
                std = torch.zeros(size=noise_size).fill_(action_var)
                noise = torch.normal(mean=torch.tensor(0.0),
                                     std=std).unsqueeze(-1)
                noise = torch.clip(noise, -noise_clip, noise_clip)
            else:
                noise = torch.normal(mean=torch.tensor(0.0),
                                     std=torch.tensor(target_var))
            action = torch.clip(policy_net(state)+noise, a_low, a_high)
            return action

    def disable_grads(neural_net):
        for parameter in neural_net.parameters():
            parameter.requires_grad=False
        return neural_net
    
    def enable_grads(neural_net):
        for parameter in neural_net.parameters():
            parameter.requires_grad=True
        return neural_net
    
    def collect_data(p_net, steps):
        s, _ = env.reset()
        for _ in range(steps):
            a = select_action(p_net, torch.as_tensor(s)).numpy()
            ns, r, te, tr, _ = env.step(a)
            d = te or tr
            replay_buffer.push(s, a, r, ns, d)
            if d:
                s, _ = env.reset()
            else:
                s = ns
    
    def update():
        for i in range(1, 51):
            data = replay_buffer.sample(batch_size)
            s, a, r, ns, t = data["state"], data["action"], data["reward"],\
                data["next_state"], data["terminal"]
            t = t.type(torch.int).unsqueeze(-1)
            r = r.unsqueeze(-1)
            a = a.unsqueeze(-1)
            with torch.no_grad():
                target_actions = select_action(p_net_target, ns, clip=True)
                targets = [r + gamma*(1-t)*q_target(ns, target_actions)
                           for q_target in q_nets_target]
            target = torch.minimum(targets[0], targets[1])
            
            for j in range(2):
                q_optimizers[j].zero_grad()
                q_loss = ((q_nets[j](s, a)-target)**2).mean()
                q_loss.backward()
                q_optimizers[j].step()
                  
            if i % 2 == 0 and i != 0:
                q_nets[0] = disable_grads(q_nets[0])
                p_optimizer.zero_grad()
                p_loss = -q_nets[0](s, p_net(s)).mean()
                p_loss.backward()
                p_optimizer.step()
                q_nets[0] = enable_grads(q_nets[0])
            
                with torch.no_grad():
                    for k in range(2):
                        for p, t in zip(q_nets[k].parameters(),
                                        q_nets_target[k].parameters()):
                            t.mul_(polyak)
                            t.add_((1-polyak)*p)
    
                    for p, t in zip(p_net.parameters(),
                                    p_net_target.parameters()):
                        t.mul_(polyak)
                        t.add_((1-polyak)*p)

    q_nets = [Q_NET(3, 32, 1) for _ in range(2)]
    q_nets_target = [disable_grads(deepcopy(net)) for net in q_nets]
    q_optimizers = [AdamW(net.parameters(), lr=1e-3) for net in q_nets]
    p_net = POLICY_NET(3, 32, 1, -2, 2)
    p_net_target = disable_grads(deepcopy(p_net))
    p_optimizer = AdamW(p_net.parameters(), lr=1e-3)
    
    replay_buffer = ReplayBuffer(3, 10000)

    collect_data(p_net_target, 10000)

    s, _ = env.reset()
    rewards_per_epoch = []
    durations = []
    reward_per_episode = 0
    episode_duration = 0
    for t in tqdm(range(1, epochs*steps_per_epoch+1)):
        a = select_action(p_net, torch.as_tensor(s), 0.1).numpy()
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
                for t in range(200):
                    test_action = select_action(p_net, torch.as_tensor(test_state)).numpy()
                    test_state, _, d1, d2, _ = test_env.step(test_action)
                    if d1 or d2:
                        test_state, _ = test_env.reset()
            test_env.close()
            rewards_per_epoch = []
            durations = []

TD3(env, epochs=20, steps_per_epoch=4000, gamma=0.99, polyak=0.995,
    batch_size=64, update_every=50, noise_clip=0.5, action_var=0.2,
    target_var=0.1)
