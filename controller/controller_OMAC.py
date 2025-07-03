import torch
import numpy as np
import rowan
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from controller.controller_PID import PIDController


torch.set_default_tensor_type('torch.DoubleTensor')


class MetaAdapt(PIDController):
    def __init__(self, pid_params=None):
        super().__init__(pid_params=pid_params)
    
    def get_residual(self, X, imu):
        q = X[3:7]
        R = rowan.to_matrix(q)

        H = self.params['m'] * np.eye(3)
        G = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + G - u)
        return y
    
    def get_Fr(self, X, imu, pd, vd, ad, meta_adapt_trigger):
        y = self.get_residual(X, imu)
        fhat_F = self.get_f_hat(X)
        self.inner_adapt(X, fhat_F, y)
        self.update_batch(X, fhat_F, y)
        if (meta_adapt_trigger and self.state=='train'):
            self.meta_adapt()
        
        Fr,Fr_dot = super().get_Fr(X, imu, pd, vd, ad, meta_adapt_trigger)
        f_hat = self.get_f_hat(X)
        return Fr-f_hat, Fr_dot
    
    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed
    
    def get_f_hat(self,X):
        raise NotImplementedError
    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError
    def update_batch(self, X, fhat, y):
        raise NotImplementedError
    def meta_adapt(self, ):
        raise NotImplementedError


class MetaAdaptLinear(MetaAdapt):
    # f_hat = (Wx+b) * A * a
    def __init__(self, 
                 pid_params=None, 
                 dim_a=100, 
                 eta_a_base=0.01, 
                 eta_A_base=0.01):
        super().__init__(pid_params=pid_params)
        self.dim_a = dim_a - dim_a % 3
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base
        self.state = 'train'
    
    def reset_controller(self):
        super().reset_controller()
        self.dim_A = dim_A = int(self.dim_a / 3)
        # setup_seed()
        self.W = np.random.uniform(low=-1, high=1, size=(dim_A, 13))
        self.a = np.random.normal(loc=0, scale=1, size=(self.dim_a))
        # self.a = np.zeros(shape=self.dim_a)
        self.b = np.random.uniform(low=-1, high=1, size=(dim_A))
        
        # self.b = np.zeros(shape=[dim_A])
        self.inner_adapt_count = 0
        self.meta_adapt_count = 0
        self.batch = []
    
    def get_Y(self, X):
        return np.kron(np.eye(3), self.W @ X + self.b)

    def get_f_hat(self, X):
        return self.get_Y(X) @ (self.a / np.linalg.norm(self.a))
    
    def update_batch(self, X, fhat, y):
        self.batch.append((X, fhat, y, self.a.copy()))
    
    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_Y(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)
        for i in range(self.dim_A):
            for X, fhat, y, a in self.batch:
                self.W[i,:] -= eta_A * 2 * (fhat - y).transpose() @ \
                    np.array([a[i],a[i+self.dim_A],a[i+self.dim_A*2]]) @ X
        self.batch = []


class MetaAdaptDeep(MetaAdapt):
    # f_hat = phi(x)^T * a
    class Phi(nn.Module):
        def __init__(self, input_kernel, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(input_kernel, layer_sizes[0]))
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
                 eta_a_base=0.001, 
                 eta_A_base=0.001):
        super().__init__(pid_params=pid_params)
        self.dim_a = dim_a - dim_a%3
        self.layer_sizes = layer_size
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base
        self.loss = nn.MSELoss()
        self.state = 'train'
    
    def reset_controller(self):
        super().reset_controller()
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(input_kernel=13, dim_kernel=self.dim_a//3, layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.inner_adapt_count = 0
        self.batch = []
    
    def get_phi(self, X):
        with torch.no_grad():
            return np.kron(np.eye(3), self.phi(torch.from_numpy(X)).numpy())
    
    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return phi @ self.a

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))
    
    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        for X, y, a in self.batch:
            phi = torch.kron(torch.eye(3), self.phi(torch.from_numpy(X)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a)), torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        self.batch = []
