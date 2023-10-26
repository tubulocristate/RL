import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.b_s = np.zeros((max_size, state_dim), dtype=np.float32)
        self.b_a = np.zeros((max_size, action_dim), dtype=np.float32)
        self.b_r = np.zeros((max_size, 1), dtype=np.float32)
        self.b_ns = np.zeros((max_size, state_dim), dtype=np.float32)
        self.b_t = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr, self.max_size = 0, max_size

    def push(self, s, a, r, ns, t):
        assert self.ptr < self.max_size
        self.b_s[self.ptr] = s
        self.b_a[self.ptr] = a
        self.b_r[self.ptr] = r
        self.b_ns[self.ptr] = ns
        self.b_t[self.ptr] = t
        self.ptr  = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indexes = np.random.randint(self.max_size, size=(batch_size))
        batch_s = self.b_s[indexes]
        batch_a = self.b_a[indexes]
        batch_r = self.b_r[indexes]
        batch_ns = self.b_ns[indexes]
        batch_t = self.b_t[indexes]
        batch = dict(s=batch_s, a=batch_a, r=batch_r,
                     ns=batch_ns, t=batch_t)
        return {k : torch.tensor(v) for k, v in batch.items()}

