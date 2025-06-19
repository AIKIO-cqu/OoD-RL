import numpy as np
import torch
import random


class ReplayBuffer:
    def __init__(self, s_dim, a_dim, capacity, batch_size, device):
        self.device = device
        self.capacity = capacity
        self.s = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False, device=self.device)
        self.a = torch.zeros(size=[capacity, a_dim], dtype=torch.float, requires_grad=False, device=self.device)
        self.s_ = torch.zeros(size=[capacity, s_dim], dtype=torch.float, requires_grad=False, device=self.device)
        self.r = torch.zeros(size=[capacity], dtype=torch.float, requires_grad=False, device=self.device)
        self.batch_size = batch_size
        self.counter = 0

    def store_transition(self, s, a, s_, r):
        index = self.counter % self.capacity
        self.s[index] = torch.tensor(s, dtype=torch.float, device=self.device)
        self.a[index] = torch.tensor(a, dtype=torch.float, device=self.device)
        self.s_[index] = torch.tensor(s_, dtype=torch.float, device=self.device)
        self.r[index] = torch.tensor(r, dtype=torch.float, device=self.device)
        self.counter += 1

    def get_sample(self):
        index = random.choices(range(min(self.capacity, self.counter)), k=self.batch_size)
        s = self.s[index]              # [batch, s_dim]
        a = self.a[index]              # [batch, a_dim]
        s_ = self.s_[index]            # [batch, s_dim]
        r = self.r[index]              # [batch]
        r = torch.unsqueeze(r, dim=-1) # [batch, 1]
        return s, a, s_, r