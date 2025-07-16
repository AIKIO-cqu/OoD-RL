import random
import argparse
import torch
import numpy as np
import json
import quadsim
import trajectory
import plot
from tqdm import tqdm
from controller import controller_TD3, controller_PPO
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


def train(C, Q, algo_name="", reset_control=True):
    print("Training " + algo_name)
    if (reset_control):
        C.reset_controller()       # 重置控制器状态
    C.reset_time()                 # 重置控制器时间
    C.train(total_timesteps=500*2000,
            eval_freq=5*2000,
            n_eval_episodes=1)
    

def test(C, Q, wind_velo, algo_name="", reset_control=True, time=None):
    print("Testing " + algo_name)
    C.state = 'test'
    if time is not None:
        C.load_model(C.best_model_dir + '/best_model.zip')  # 加载最佳模型
    ace_error_list = np.empty(10)

    for round in range(10):
        setup_seed(456+round*11)
        Wind_Velocity = np.random.uniform(low=-wind_velo, high=0., size=(20,3))
        t_readout = -0.0
        p_list = []
        pd_list = []
        reward_list = []
        imu_meas = np.zeros(3)

        obs, info = Q.reset(wind_velocity_list=Wind_Velocity)
        t, X = info['t'], info['X']
        if (reset_control):
            C.reset_controller()       # 重置控制器状态
        C.reset_time()                 # 重置控制器时间

        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']), 
                    desc=f"Round {round+1}/10", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        while t < Q.params['t_stop']:
            pd, vd, ad = Q.get_desired(t)
            action = C.get_action(obs=obs, t=t, pd=pd, vd=vd, ad=ad, imu=imu_meas, t_last_wind_update=Q.t_last_wind_update)
            obs, reward, terminated, truncated, info = Q.step(action)
            reward_list.append(reward)
            t, Xdot, Z, X = info['t'], info['Xdot'], info['Z'], info['X']
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
        plot.plot_traj(p_list, pd_list, algo_name, time)
        plot.plot_reward(reward_list, algo_name, time)
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    print("*******", algo_name, "*******")
    print("ACE Error: %.3f(%.3f)" % (ace, std))
    return np.mean(ace_error_list)


def run_TD3(traj, wind_velo):
    # ************** TD3 **************
    q_td3 = quadsim.Quadrotor(traj=traj)
    c_td3 = controller_TD3.TD3Agent(q_td3)
    # ********* Train && Test *********
    train(c_td3, q_td3, "TD3")
    test(c_td3, q_td3, wind_velo, "TD3", reset_control=True, time=c_td3.time)
    # *********** Only Test ***********
    # c_td3.load_model("model/TD3/2025-07-15_13-36/best_model.zip")
    # test(c_td3, q_td3, wind_velo, "TD3", reset_control=True, time=None)

def run_PPO(traj, wind_velo):
    # ************** PPO **************
    q_ppo = quadsim.Quadrotor(traj=traj)
    c_ppo = controller_PPO.PPOAgent(q_ppo)
    # ********* Train && Test *********
    train(c_ppo, q_ppo, "PPO")
    test(c_ppo, q_ppo, wind_velo, "PPO", reset_control=True, time=c_ppo.time)
    # *********** Only Test ***********
    # c_ppo.load_model("model/PPO/2025-07-06_05-11/best_model.zip")
    # test(c_ppo, q_ppo, wind_velo, "PPO", reset_control=True, time=None)


parser = argparse.ArgumentParser()
if __name__ == '__main__':
    parser.add_argument('--logs', type=int, default=1)
    parser.add_argument('--trace', type=str, default='hover')
    parser.add_argument('--wind', type=str, default='empty')  # 默认为无风测试
    parser.add_argument('--use_bayes', type=bool, default=False)
    args = parser.parse_args()
    if (args.wind=='breeze'):
        wind_velo = 4
    elif (args.wind=='strong_breeze'):
        wind_velo = 8
    elif (args.wind=='gale'):
        wind_velo = 12
    elif (args.wind=='empty'):
        wind_velo = 0
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
    
    run_TD3(traj, wind_velo)

    # traj = trajectory.hover()
    # run_TD3(traj, wind_velo)
    # traj = trajectory.sin_forward()
    # run_TD3(traj, wind_velo)
    # traj = trajectory.fig8()
    # run_TD3(traj, wind_velo)
    # traj = trajectory.spiral_up()
    # run_TD3(traj, wind_velo)

    # run_PPO(traj, wind_velo)