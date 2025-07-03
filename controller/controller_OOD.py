import torch
import numpy as np
from controller.controller_OMAC import MetaAdaptDeep


torch.set_default_tensor_type('torch.DoubleTensor')


class MetaAdaptOoD(MetaAdaptDeep):
    def __init__(self, 
                 pid_params=None, 
                 dim_a=100, 
                 layer_size=(25,30), 
                 eta_a_base=0.001,
                 eta_A_base=0.001, 
                 noise_x=0.01, 
                 noise_a=0.01):
        super().__init__(pid_params=pid_params,
                         dim_a=dim_a, 
                         layer_size=layer_size, 
                         eta_a_base=eta_a_base, 
                         eta_A_base=eta_A_base)
        self.noise_x = noise_x
        self.noise_a = noise_a
    
    def inner_adapt(self, X, fhat, y):
        self.a -= self.eta_a_base * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def meta_adapt(self):
        self.optimizer.zero_grad()

        loss = 0
        for X, y, a in self.batch:
            X = X + self.noise_x*np.random.normal(0,1,X.shape)
            a = a + self.noise_a*np.random.normal(0,1,a.shape)
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        self.batch = []