import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from controller.controller_OMAC import MetaAdaptDeep


torch.set_default_tensor_type('torch.DoubleTensor')


class NeuralFly(MetaAdaptDeep):
    class H(nn.Module):
        def __init__(self, start_kernel, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(start_kernel, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self,
                 pid_params=None, 
                 dim_a=100, 
                 layer_size=(25,30), 
                 eta_a_base=0.01, 
                 eta_A_base=0.001):
        super().__init__(pid_params=pid_params,
                         dim_a=dim_a, 
                         layer_size=layer_size, 
                         eta_a_base=eta_a_base, 
                         eta_A_base=eta_A_base)
        self.wind_idx = 0
        self.alpha = 0.5
        # self.lr = lr

    def reset_controller(self):
        super().reset_controller()
        self.h = self.H(start_kernel = self.dim_a//3, dim_kernel=3, layer_sizes=self.layer_sizes)
        self.h_optimizer = optim.Adam(params=self.h.parameters(), lr=0.1)
        self.h_loss = nn.CrossEntropyLoss()
    
    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        target = torch.tensor([self.wind_idx], dtype=int)
        for X, y, a in self.batch:
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y)) \
                 - self.alpha * self.h_loss(self.h(self.phi(torch.from_numpy(X))).unsqueeze(0), 
                                            target).detach()
        loss.backward()
        self.optimizer.step()

        if (np.random.uniform(0,1) < 0.5):
            loss_h = 0
            self.h_optimizer.zero_grad()
            for X, y, a in self.batch:
                phi = self.phi(torch.from_numpy(X)).detach()
                h = self.h(phi)
                loss_h += self.h_loss(h.unsqueeze(0), target)
            loss_h.backward()
            self.h_optimizer.step()
        
        self.batch = []
