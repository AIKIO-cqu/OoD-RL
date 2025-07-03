import random
import argparse
import torch
import numpy as np
import json
import quadsim
import trajectory
from tqdm import tqdm
from controller import controller_TD3
from stable_baselines3.common.env_checker import check_env


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(C, Q, traj, algo_name="", reset_control=True):
    print("Training " + algo_name)
    
    check_env(Q)
    # Wind_Velocity = np.zeros((20, 3))  # 无风测试
    obs, info = Q.reset()
    
    C.set_env(Q)
    if (reset_control):
        C.reset_controller()
    C.reset_time()

    C.train(total_timesteps=2000)
    

def test(C, Q, traj, algo_name, reset_control=True):
    print("Testing " + algo_name)
    C.state = 'test'
    ace_error_list = np.empty(10)

    for round in range(10):
        setup_seed(456+round*11)
        # Wind_Velocity = np.random.uniform(low=-Wind_velo, high=0., size=(20,3))
        Wind_Velocity = np.zeros((20, 3))  # 无风测试
        t_readout = -0.0
        p_list = []
        pd_list = []
        imu_meas = np.zeros(3)

        obs, info = Q.reset(wind_velocity_list=Wind_Velocity)
        X = obs[0: 13]
        t = info['t']
        if (reset_control):
            C.reset_controller()       # 重置控制器状态
        C.reset_time()                 # 重置控制器时间

        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']), 
                    desc=f"Round {round+1}/10", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        while t < Q.params['t_stop']:
            pd, vd, ad = traj(t)
            action = C.get_action(obs=obs, t=t, pd=pd, vd=vd, ad=ad, imu=imu_meas, t_last_wind_update=Q.t_last_wind_update)
            obs, reward, terminated, truncated, info = Q.step(action)
            X = obs[0: 13]
            t, Xdot, Z = info['t'], info['Xdot'], info['Z']
            imu_meas = Xdot[7:10]
            if t>=t_readout:
                p_list.append(X[0:3])
                pd_list.append(pd)
                t_readout += Q.params['dt_readout']
                time_percentage = (t / Q.params['t_stop']) * 100
                pbar.update(1)
                pbar.set_description(f"Round {round+1}/10 Time:{time_percentage:5.1f}%")
        pbar.close()
        squ_error = np.sum((np.array(p_list)-np.array(pd_list))**2, 1)
        ace_error = np.mean(np.sqrt(squ_error))
        print("Round %d: ACE Error: %.3f" % (round + 1, ace_error))
        ace_error_list[round] = ace_error
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    print("*******", algo_name, "*******")
    print("ACE Error: %.3f(%.3f)" % (ace, std))
    return np.mean(ace_error_list)


parser = argparse.ArgumentParser()
if __name__ == '__main__':
    parser.add_argument('--logs', type=int, default=1)
    parser.add_argument('--trace', type=str, default='hover')
    parser.add_argument('--wind', type=str, default='gale')
    parser.add_argument('--use_bayes', type=bool, default=False)
    args = parser.parse_args()
    if (args.wind=='breeze'):
        Wind_velo = 4
    elif (args.wind=='strong_breeze'):
        Wind_velo = 8
    elif (args.wind=='gale'):
        Wind_velo = 12
    elif (args.wind=='empty'):
        Wind_velo = 0
    else:
        raise NotImplementedError

    if (args.trace=='hover'):
        traj = trajectory.hover()
    elif (args.trace=='fig8'):
        traj = trajectory.fig8()
    elif (args.trace=='spiral'):
        traj = trajectory.spiral_up()
    elif (args.trace=='sin'):
        traj = trajectory.sin_forward()
    else:
        raise NotImplementedError
    
    c_td3 = controller_TD3.TD3Agnet()
    q_td3 = quadsim.Quadrotor()
    train(c_td3, q_td3, traj, "TD3")
    test(c_td3, q_td3, traj, "TD3", reset_control=True)