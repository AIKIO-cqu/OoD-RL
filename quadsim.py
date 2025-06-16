import torch
import numpy as np
import random
import json
import rowan
from tqdm import tqdm
import pkg_resources
import copy
import gymnasium as gym
from gymnasium import spaces
import trajectory
from collections import deque

DEFAULT_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/px4.json')

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Quadrotor(gym.Env):
    def __init__(self, pid_controller=None, traj=None, name='', state='train', paramsfile=DEFAULT_PARAMETER_FILE, px4paramfile=DEFAULT_PX4_PARAM_FILE, **kwargs):
        super(Quadrotor, self).__init__()

        # Quadrotor params
        self.params = readparamfile(paramsfile)
        self.params.update(kwargs)

        # PX4 params
        self.px4_params = readparamfile(px4paramfile)
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'],
                                                  self.px4_params['MC_PITCHRATE_P'],
                                                  self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'],
                                                  self.px4_params['MC_PITCHRATE_I'],
                                                  self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'],
                                                  self.px4_params['MC_PITCHRATE_D'],
                                                  self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'],
                                                  self.px4_params['MC_PITCHRATE_K'],
                                                  self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'],
                                                   self.px4_params['MC_PR_INT_LIM'],
                                                   self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])

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

        # self.pid_controller = pid_controller  # PID控制器实例

        self.traj = traj
        self.name = name
        self.state = state

        self.thrust_max = 4 * self.params['C_T'] * self.params['motor_max_speed'] ** 2
        self.angrate_max = self.px4_params['angrate_max']
        
        self.action_space = spaces.Box(
            low=np.array([-self.params['g'], -self.angrate_max[0], -self.angrate_max[1], -self.angrate_max[2]], dtype=np.float32),
            high=np.array([2*self.params['g'], self.angrate_max[0], self.angrate_max[1], self.angrate_max[2]], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.obs_cfg = {
            'future_len': 10,  # 未来参考轨迹长度
            'history_len': 5   # 历史观测长度
        }
        self.history_len = self.obs_cfg['history_len']
        self.init_history()
        obs_dim = self.calculate_obs_dimension()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def init_history(self):
        if self.history_len > 0:
            self.history_buffer = deque(maxlen=self.history_len)
    
    def calculate_obs_dimension(self):
        future_obs_dim = self.obs_cfg['future_len'] * 3  # 未来参考轨迹维度
        history_obs_dim = self.obs_cfg['history_len'] * 3 # 历史观测维度(pos)
        history_obs_dim += 4 # 历史观测维度(action)
        history_obs_dim += 3 # 历史观测维度(ref_err)
        obs_dim = future_obs_dim + history_obs_dim
        return obs_dim

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = random.randint(0, 10000)
        print("seed:", seed)
        setup_seed(seed)  # 设置所有随机数生成器的种子
        if self.state=='test' and options is not None:
            Wind_Velocity = np.random.uniform(low=-options['wind_velo'], high=0., size=(20,3))  # 测试风
        else:
            # Wind_Velocity = np.random.gamma(shape=1., scale=0.5*(seed%3+1), size=(20,3))      # 训练风
            Wind_Velocity = np.zeros((20, 3))
        
        print("Trajectory:", self.traj.name)

        # self.pid_controller.reset_controller()  # 重置PID控制器状态
        # self.pid_controller.reset_time()        # 重置PID控制器时间

        self.t_last_wind_update = 0             # 上次更新风速的时间
        self.wind_count = 0                     # 风速列表的索引
        self.VwindList = Wind_Velocity          # 风速列表

        X = np.zeros(13)                        # 初始化状态向量
        X[3] = 1.                               # 初始化四元数为单位四元数
        # X[0:3] = np.random.uniform(low=-5, high=5, size=(3,))  # 初始化随机位置
        # print(f"初始位置: {X[0:3]}")

        hover_motor_speed = np.sqrt(self.params['m'] * self.params['g'] / (4 * self.params['C_T']))
        Z = hover_motor_speed * np.ones(4)      # 初始化电机转速
        self.X = X
        self.Z = Z
        self.t = self.params['t_start']
        
        self.imu = np.zeros(3)                  # 初始化IMU测量值

        self.w_error_int = np.zeros(3)          # 角速度误差积分
        self.w_filtered = np.zeros(3)           # 一阶低通滤波器的状态
        self.w_filtered_last = np.zeros(3)      # 上一时刻的滤波状态

        if hasattr(self, 'history_buffer'):
            self.history_buffer.clear()          # 清空历史观测缓冲区
            for _ in range(self.history_len):
                self.history_buffer.append(X[0:3].copy())  # 初始化历史观测为当前状态位置
        self.last_action = np.zeros(4)

        obs = self.get_observation()
        return obs.astype(np.float32), {'t': self.t}

    def get_observation(self):
        all_obs = []

        # 1. future reference trajectory
        future_obs = self.get_future_obs(self.obs_cfg['future_len'])
        all_obs.append(future_obs)

        # # 2. current state
        # X_obs = X.copy()  # 复制状态向量
        # if self.state == "train":
        #     noise_mask = np.random.random(X_obs.shape) < 0.5  # 50% 概率添加噪声
        #     X_obs += 0.01 * np.random.normal(0, 1, X_obs.shape) * noise_mask
        #     X_obs[3:7] /= np.linalg.norm(X_obs[3:7])  # 四元数归一化
        # all_obs.append(X_obs)

        # 3. history observation
        pd, vd, ad = self.traj(self.t)
        history_obs = self.get_history_obs(self.obs_cfg['history_len'], pd)
        all_obs.append(history_obs)

        return np.concatenate(all_obs, axis=-1)

    def get_future_obs(self, future_len):
        future_obs = []
        dt = self.params['dt']
        for i in range(future_len):
            # 计算未来时刻的参考轨迹
            future_t = self.t + (i + 1) * dt
            pd, vd, ad = self.traj(future_t)
            # 计算相对位置
            current_pos = self.X[0:3]
            relative_pos = pd - current_pos
            future_obs.append(relative_pos)
        future_obs = np.array(future_obs).flatten()
        return future_obs

    def get_history_obs(self, history_len, pd):
        if not hasattr(self, 'history_buffer') or len(self.history_buffer) == 0:
            return np.zeros(history_len*3 + 4 + 3)  # 5个位置 + 1个动作 + 1个误差
        
        hist_pos_list = []
        buffer_list = list(self.history_buffer)
        for i in range(history_len):
            if i < len(buffer_list):
                pos = buffer_list[-(i+1)]
            else:
                pos = buffer_list[0] if len(buffer_list)>0 else np.zeros(3)
            hist_pos_list.append(pos)
        hist_pos = np.array(hist_pos_list).flatten()

        last_action = self.last_action if hasattr(self, 'last_action') else np.zeros(4)
        current_ref_err = self.X[0:3] - pd
        
        return np.concatenate([hist_pos, last_action, current_ref_err])

    def f(self, X, Z, t, test=False):
        p = X[0:3]
        X[3:7] = rowan.normalize(X[3:7])
        q = X[3:7]
        R = rowan.to_matrix(q)
        v = X[7:10]
        w = X[10:]

        #print("Z", Z)
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
            Z = alpha_m*u + (1 - alpha_m) * Z

        return np.maximum(np.minimum(Z, self.params['motor_max_speed']), self.params['motor_min_speed'])

    def limit(self, array, upper_limit, lower_limit=None):
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]

    def mixer(self, torque_sp, T_sp):
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        return omega
    
    def getu(self, action):
        # action: [a_z, omega_x, omega_y, omega_z]
        T_sp = np.clip(action[0] * self.params['m'], 0, self.thrust_max)    # 期望总推力
        w_sp = action[1:4]                                                  # 期望角速度
        
        dt = self.params['dt']
        w=self.X[10:]

        w_error = w_sp - w
        self.w_error_int += dt * w_error
        self.limit(self.w_error_int, self.px4_params['angrate_int_lim'])

        const_w_filter = np.exp(-dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w

        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        self.w_filtered_last[:] = self.w_filtered[:]  # Python is a garbage language

        alpha_sp = self.px4_params['angrate_gain_K'] \
                    @ (self.px4_params['angrate_gain_P'] @ w_error
                       + self.px4_params['angrate_gain_I'] @ self.w_error_int
                       - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)
        self.limit(alpha_sp, self.px4_params['angacc_max'])

        u = self.mixer(torque_sp=alpha_sp, T_sp=T_sp)  # 将期望力矩和期望总推力转换为电机控制输入 u

        return u
        
    
    def step(self, action):
        u = self.getu(action) # 电机控制输入
        
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
        self.imu = Xdot[7:10]

        if hasattr(self, 'history_buffer'):
            self.history_buffer.append(X[0:3].copy())

        reward, terminated, truncated = self.calculate_reward(X, action, t)
        self.last_action = action.copy()  # 保存上一个动作

        obs = self.get_observation()
        return obs.astype(np.float32), reward, terminated, truncated, {'t': self.t, 'Z': self.Z}

    def calculate_reward(self, X, action, t):
        p = X[0:3]
        v = X[7:10]
        q = X[3:7]
        w = X[10:]
        pd, vd, ad = self.traj(t)

        reward_smooth = 0.5 * np.exp(-np.linalg.norm(action-self.last_action))  # 平滑奖励
        reward_min = 0.5 * np.exp(-np.linalg.norm(action))                      # 最小化动作奖励
        reawrd_pos = 1.0 * np.exp(-np.linalg.norm(p-pd))                        # 位置奖励
        reward_yaw = 0.2 * np.exp(-np.abs(0.0-rowan.to_euler(q)[2]))            # 偏航奖励
        reward_vel = 0.1 * np.exp(-np.linalg.norm(v-vd))                        # 速度奖励

        reward = reward_smooth + reward_min + reawrd_pos + reward_yaw + reward_vel

        terminated = False
        truncated = False
        if self.t >= self.params['t_stop'] / 4:
            terminated = True
            print(f"达到最大仿真时间: {self.t:.2f}，终止仿真")
        return reward, terminated, truncated
    
    def get_residual(self, X, imu, motor_speed):
        q = X[3:7]
        R = rowan.to_matrix(q)

        H = self.params['m'] * np.eye(3)
        G = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(motor_speed ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + G - u)      # ma = u + y - G

        return y
    
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

    # def generate_traj(self):
    #     trace = random.choice(['hover', 'fig8', 'spiral', 'sin'])
    #     if (trace=='hover'):
    #         traj = trajectory.hover()
    #     elif (trace=='fig8'):
    #         traj = trajectory.fig8()
    #     elif (trace=='spiral'):
    #         traj = trajectory.spiral_up()
    #     elif (trace=='sin'):
    #         traj = trajectory.sin_forward()
    #     else:
    #         raise NotImplementedError
    #     return traj