import numpy as np
import json
import rowan
import pkg_resources
import gymnasium as gym


DEFAULT_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Quadrotor(gym.Env):
    def __init__(self, traj, paramsfile=DEFAULT_PARAMETER_FILE, **kwargs):
        super(Quadrotor, self).__init__()
        self.traj = traj
        
        # Quadrotor params
        self.params = readparamfile(paramsfile)
        self.params.update(kwargs)

        # 控制分配矩阵
        # |C_T           C_T        C_T         C_T      |   计算总推力
        # |-C_T*l_arm   -C_T*l_arm  C_T*l_arm   C_T*l_arm|   计算横滚力矩
        # |-C_T*l_arm    C_T*l_arm  C_T*l_arm  -C_T*l_arm|   计算俯仰力矩
        # |-C_q          C_q       -C_q         C_q      |   计算偏航力矩
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])
        
        self.params['J'] = np.array(self.params['J'])   # 惯性矩阵
        self.Jinv = np.linalg.inv(self.params['J'])     # 惯性矩阵的逆
        #self.params['process_noise_covariance'] = np.array(self.params['process_noise_covariance'])
        l_arm = self.params['l_arm']    # 机臂长度
        h = self.params['h']            # 机身高度

        # 四个电机相对于无人机质心的位置，每行对应一个电机的(x,y,z)坐标
        self.r_arms = np.array(((l_arm, -l_arm, h),
                                (-l_arm, -l_arm, h),
                                (-l_arm, l_arm, h),
                                (l_arm, l_arm, h)))
        
        self.Vwind = np.zeros(3)  # 风速

        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(13,), 
            dtype=np.float64
        )

        self.max_thrust = 4 * self.params['C_T'] * self.params['motor_max_speed'] ** 2
        self.max_angrate = np.array((self.params['MC_ROLLRATE_MAX'],
                                     self.params['MC_PITCHRATE_MAX'],
                                     self.params['MC_YAWRATE_MAX']))
        self.action_low = np.array([0.0, -self.max_angrate[0], -self.max_angrate[1], -self.max_angrate[2]])
        self.action_high = np.array([self.max_thrust / self.params['m'],
                                     self.max_angrate[0],
                                     self.max_angrate[1],
                                     self.max_angrate[2]])
        # RL
        # self.action_space = gym.spaces.Box(
        #     low=np.array([0.0, -self.max_angrate[0], -self.max_angrate[1], -self.max_angrate[2]]),
        #     high=np.array([self.max_thrust/self.params['m'], self.max_angrate[0], self.max_angrate[1], self.max_angrate[2]]),
        #     shape=(4,),
        #     dtype=np.float64
        # )
        # RL_PID
        self.action_space = gym.spaces.Box(
            low=np.array([-9.81, -3.14, -3.14, -3.14]),
            high=np.array([9.81, 3.14, 3.14, 3.14]),
            shape=(4,),
            dtype=np.float64
        )

    def reset(self, seed=None, wind_velocity_list=None):
        self.t_last_wind_update = 0             # 上次更新风速的时间
        self.wind_count = 0                     # 风速列表的索引
        # 如果是调用sb3库的强化学习算法进行训练，需要直接在环境中定义好风速（无风或gamma分布的风）
        if wind_velocity_list is None:
            wind_velocity_list = np.zeros((20, 3))
        self.VwindList = wind_velocity_list     # 风速列表
        X = np.zeros(13)                        # 初始化状态向量
        X[3] = 1.
        # X[0:3] = np.random.uniform(low=-1.0, high=1.0, size=3)  # 随机初始化位置
        hover_motor_speed = np.sqrt(self.params['m'] * self.params['g'] / (4 * self.params['C_T']))
        Z = hover_motor_speed * np.ones(4)      # 初始化电机转速
        self.X = X
        self.Z = Z
        self.t = self.params['t_start']

        obs = self.get_observation()
        info = {'t': self.t, 'X': self.X}
        return obs, info

    def get_observation(self):
        pd, vd, ad = self.get_desired(self.t)
        obs = self.X.copy()
        obs[0:3] -= pd
        obs[7:10] -= vd
        return obs
    
    def f(self, X, Z, t, test=False):
        p = X[0:3]
        X[3:7] = rowan.normalize(X[3:7])
        q = X[3:7]
        R = rowan.to_matrix(q)
        v = X[7:10]
        w = X[10:]

        T, *tau_mec = self.B @ (Z ** 2)     # 使用分配矩阵 B 将电机转速平方映射到总推力和三轴力矩
        Xdot = np.empty(13)
        Xdot[0:3] = v
        Xdot[3:7] = rowan.calculus.derivative(q, w)

        F_mec = np.empty(3)
        F_mec = T * (R @ np.array([0., 0., 1.])) - np.array([0., 0., self.params['g']*self.params['m']])    # 机械力
        F_wind, tau_wind = self.get_wind_effect(X, Z, t)    # 风力，风力矩
        Xdot[7:10] = (F_mec + F_wind) / self.params['m']
        Xdot[10:] = np.linalg.solve(self.params['J'], np.cross(self.params['J'] @ w, w) + tau_mec + tau_wind)
        return Xdot
    
    def update_motor_speed(self, u, dt, Z=None):
        if Z is None:
            Z = u
        else:
            # 使用一阶低通滤波器模拟电机响应
            alpha_m = 1 - np.exp(-self.params['w_m'] * dt)
            # Z = alpha_m*u + (1 - alpha_m) * Z
            Z = alpha_m*u + (1 - alpha_m) * Z
        return np.maximum(np.minimum(Z, self.params['motor_max_speed']), self.params['motor_min_speed'])

    def limit(self, array, upper_limit, lower_limit=None):
        upper_limit = np.array(upper_limit)
        if lower_limit is None:
            lower_limit = - np.array(upper_limit)
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]

    def mixer(self, torque_sp, T_sp):
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        return omega
    
    def get_u(self, action):
        # action: [a_z, omega_x, omega_y, omega_z]
        T_sp = np.clip(action[0] * self.params['m'], 0, self.max_thrust)    # 期望总推力
        w_sp = action[1:4]                                                  # 期望角速度
        w = self.X[10:]
        w_error = w_sp - w
        alpha_sp = np.diag([1,1,1]) @ w_error + np.cross(w.T, np.dot(self.params['J'], w).T).T
        self.limit(alpha_sp, self.params['angacc_max'])
        u = self.mixer(torque_sp=alpha_sp, T_sp=T_sp)  # 将期望力矩和期望总推力转换为电机控制输入 u
        return u

    def step(self, action):
        u = self.get_u(action)

        X = self.X
        t = self.t
        Z = self.Z
        dt = self.params['dt']
        if self.params['integration_method'] == 'rk4':
            # RK4 method
            Z = self.update_motor_speed(Z=Z, u=u, dt=0.0)
            k1 = dt * self.f(X, Z, t)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k2 = dt * self.f(X + k1/2, Z, t+dt/2)
            k3 = dt * self.f(X + k2/2, Z, t+dt/2)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k4 = dt * self.f(X + k3, Z, t+dt)
            
            Xdot = (k1 + 2*k2 + 2*k3 + k4)/6 / dt
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6
        elif self.params['integration_method'] == 'euler':
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt)        # 先更新电机转速
            Xdot = self.f(X, Z, t)                              # 计算导数
            X = X + dt * Xdot                                   # 直接使用导数更新状态
        else:
            raise NotImplementedError
        X[3:7] = rowan.normalize(X[3:7])                        # 四元数归一化

        # noise = np.random.multivariate_normal(np.zeros(6), self.params['process_noise_covariance'])
        # X[7:] += noise
        # logentry['process_noise'] = noise

        self.X = X
        self.Z = Z
        self.t = t + dt

        obs = self.get_observation()
        reward, terminated, truncated = self.calculate_reward(X, t)
        info = {'t':self.t, 'Xdot':Xdot, 'Z':self.Z, 'X':self.X}
        return obs, reward, terminated, truncated, info
    
    def get_desired(self, t):
        pd, vd, ad = self.traj(t)
        return pd, vd, ad

    def calculate_reward(self, X, t):
        p = X[0:3]
        v = X[7:10]
        q = X[3:7]
        w = X[10:]

        pd, vd, ad = self.get_desired(t)

        pos_error = np.linalg.norm(p - pd)
        yaw_error = np.abs(0.0 - rowan.to_euler(q)[2])
        vel_error = np.linalg.norm(v - vd)
        # reward = 1.0 * -pos_error + 0.2 * -yaw_error + 0.1 * -vel_error
        reward = np.exp(-pos_error) + 0.2 * np.exp(-yaw_error) + 0.1 * np.exp(-vel_error)

        terminated = False
        truncated = False
        if self.t >= self.params['t_stop']:
            terminated = True

        return reward, terminated, truncated
        

    def get_wind_v(self, t):
        dt = t - self.t_last_wind_update    # 计算自上次风速更新以来经过的时间
        if dt > self.params['wind_update_period']:
            self.Vwind = self.VwindList[self.wind_count]
            self.wind_count += 1
            self.t_last_wind_update = t
        return self.Vwind
    
    def get_wind_effect(self, X, Z, t):
        Vwind = self.get_wind_v(t)
        v = X[7:10]
        R_wtob = rowan.to_matrix(X[3:7]).transpose()
        Vinf = Vwind - v        # 相对风速
        Vinf_B = R_wtob @ Vinf  # 转换到机体坐标系
        Vz_B = np.array((0., 0., Vinf_B[2]))    # 垂直风速
        Vs_B = Vinf_B - Vz_B    # 水平风速
        if np.linalg.norm(Vs_B) > 1e-4:
            # 计算攻角
            aoa = np.arcsin(np.linalg.norm(Vz_B)/np.linalg.norm(Vinf_B))
            # 计算电机转速
            n = np.sqrt(Z / self.params['C_T'])
            # 计算每个螺旋桨的侧向力
            Fs_per_prop = self.params['C_s'] * self.params['rho'] * (n ** self.params['k1']) \
                     * (np.linalg.norm(Vinf) ** (2 - self.params['k1'])) * (self.params['D'] \
                     ** (2 + self.params['k1'])) * ((np.pi / 2) ** 2 - aoa ** 2) \
                     * (aoa + self.params['k2'])
            # 计算总侧向力（机体坐标系）
            Fs_B = (Vs_B/np.linalg.norm(Vs_B)) * sum(Fs_per_prop)
            
            # 计算侧向力产生的力矩
            tau_s = np.zeros(3)
            for i in range(4):
                tau_s += np.cross(self.r_arms[i], (Vs_B/np.linalg.norm(Vs_B)) * Fs_per_prop[i])
        else:
            Fs_B = np.zeros(3)
            tau_s = np.zeros(3)
        Fs = R_wtob.transpose() @ Fs_B  # 将机体坐标系中的侧向力转换回世界坐标系
        return Fs, tau_s                # 返回计算出的侧向力（世界坐标系）和力矩（机体坐标系）
    