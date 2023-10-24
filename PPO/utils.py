import torch
import numpy as np

class SetOfTrajectories:
    def __init__(self, observation_dim, action_dim, max_size,
                 gamma=0.99, lam=0.97):
        self.buff_obs = np.zeros((max_size, observation_dim), dtype=np.float32)
        self.buff_act = np.zeros((max_size, action_dim), dtype=np.float32)
        self.buff_rew = np.zeros((max_size, ), dtype=np.float32)
        self.buff_ret = np.zeros((max_size, ), dtype=np.float32)
        self.buff_val = np.zeros((max_size, ), dtype=np.float32)
        self.buff_adv = np.zeros((max_size, ), dtype=np.float32)
        self.start_ptr, self.end_ptr, self.max_size = 0, 0, max_size
        self.gamma, self.lam = gamma, lam

    def push(self, obs, act, rew, val):
        assert self.end_ptr < self.max_size
        self.buff_obs[self.end_ptr] = obs
        self.buff_act[self.end_ptr] = act
        self.buff_rew[self.end_ptr] = rew
        self.buff_val[self.end_ptr] = val
        self.end_ptr += 1

    def exp_cumsum(self, sequence, discount):
        result = [0 for _ in range(len(sequence))]
        for t in reversed(range(len(sequence))):
            result_next = result[t+1] if t+1 < len(sequence) else 0
            result[t] = sequence[t] + discount*result_next
        return result

    def finish_trajectory(self, last_value):
        buff_slice = slice(self.start_ptr, self.end_ptr) 
        
        values = np.append(self.buff_val[buff_slice], last_value)
        rewards = np.append(self.buff_rew[buff_slice], last_value)

        temporal_diff = rewards[:-1] + self.gamma*values[1:] - values[:-1]
        advantages = self.exp_cumsum(temporal_diff, self.gamma*self.lam)
        self.buff_adv[buff_slice] = advantages
        returns = self.exp_cumsum(rewards, self.gamma)[:-1]
        self.buff_ret[buff_slice] = returns

        self.start_ptr = self.end_ptr

    def get(self):
        assert self.end_ptr == self.max_size
        self.start_ptr, self.end_ptr = 0, 0
        trajectories = dict(s=self.buff_obs, a=self.buff_act,
                            g=self.buff_ret, A=self.buff_adv)
        return {k : torch.as_tensor(v)
                for k, v in trajectories.items()}


