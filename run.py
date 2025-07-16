import random
import argparse
import torch
import numpy as np
import json
import quadsim
from controller import controller_PID, controller_OMAC, controller_NF, controller_OOD
import trajectory
from tqdm import tqdm
import plot
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
    ace_error_list = np.empty(3)

    for round in range(3):
        setup_seed(round)
        if (algo_name == 'Neural-Fly'):
            C.wind_idx = round
        Wind_Velocity = np.random.gamma(shape=1., scale=0.5*(round%3+1), size=(20,3))
        t_readout = -0.0
        p_list = []
        pd_list = []
        imu_meas = np.zeros(3)

        obs, info = Q.reset(wind_velocity_list=Wind_Velocity)
        t, X = info['t'], info['X']
        if (reset_control):
            C.reset_controller()       # 重置控制器状态
        C.reset_time()                 # 重置控制器时间

        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']), 
                    desc=f"Round {round+1}/3", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        while t < Q.params['t_stop']:
            pd, vd, ad = Q.get_desired(t)
            action = C.get_action(obs=obs, t=t, pd=pd, vd=vd, ad=ad, imu=imu_meas, t_last_wind_update=Q.t_last_wind_update)
            obs, reward, terminated, truncated, info = Q.step(action)
            t, Xdot, Z, X = info['t'], info['Xdot'], info['Z'], info['X']
            imu_meas = Xdot[7:10]
            if t>=t_readout:
                p_list.append(X[0:3])
                pd_list.append(pd)
                t_readout += Q.params['dt_readout']
                time_percentage = (t / Q.params['t_stop']) * 100
                pbar.update(1)
                pbar.set_description(f"Round {round+1}/3 Time:{time_percentage:5.1f}%")
        pbar.close()
        squ_error = np.sum((np.array(p_list)-np.array(pd_list))**2, 1)
        ace_error = np.mean(np.sqrt(squ_error))
        print("Round %d: ACE Error: %.3f" % (round + 1, ace_error))
        ace_error_list[round] = ace_error
    print("Training Error: ", np.mean(ace_error_list))
    return np.mean(ace_error_list)


def test(C, Q, wind_velo, algo_name="", reset_control=True, time=None):
    print("Testing " + algo_name)
    C.state = 'test'
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
        # plot.plot_traj(p_list, pd_list, algo_name, time)
        # plot.plot_reward(reward_list, algo_name, time)
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    print("*******", algo_name, "*******")
    print("ACE Error: %.3f(%.3f)" % (ace, std))
    return np.mean(ace_error_list)


def contrast_algo(best_p, traj, wind_velo):
    # ************** PID **************
    c_pid = controller_PID.PIDController(pid_params=best_p)
    q_pid = quadsim.Quadrotor(traj=traj)
    check_env(q_pid)
    test(c_pid, q_pid, wind_velo, "PID")
    # ************* Linear *************
    c_linear = controller_OMAC.MetaAdaptLinear(pid_params=best_p)
    q_linear = quadsim.Quadrotor(traj=traj)
    test(c_linear, q_linear, wind_velo, "Linear")
    # ************** OMAC **************
    c_deep = controller_OMAC.MetaAdaptDeep(pid_params=best_p, 
                                           eta_a_base=0.01, 
                                           eta_A_base=0.05)
    q_deep = quadsim.Quadrotor(traj=traj)
    train(c_deep, q_deep, "OMAC(deep)")
    test(c_deep, q_deep, wind_velo, "OMAC(deep)", False)
    # *********** Neural-Fly ***********
    c_NF = controller_NF.NeuralFly(pid_params=best_p)
    q_NF = quadsim.Quadrotor(traj=traj)
    train(c_NF, q_NF, "Neural-Fly")
    test(c_NF, q_NF, wind_velo, "Neural-Fly", False)
    # ************** OOD **************
    c_ood = controller_OOD.MetaAdaptOoD(pid_params=best_p,
                                        eta_a_base=0.01, 
                                        eta_A_base=0.05, 
                                        noise_a=0.06, 
                                        noise_x=0.06)
    q_ood = quadsim.Quadrotor(traj=traj)
    train(c_ood, q_ood, "OoD-Control")
    test(c_ood, q_ood, wind_velo, "OoD-Control", False)


parser = argparse.ArgumentParser()
if __name__ == '__main__':
    parser.add_argument('--logs', type=int, default=1)
    parser.add_argument('--trace', type=str, default='hover')
    parser.add_argument('--wind', type=str, default='gale')
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
    
    best_p = readparamfile('params/pid.json')

    contrast_algo(best_p, traj, wind_velo)

    # traj = trajectory.hover()
    # print(f"================{traj.name}================")
    # contrast_algo(best_p, traj, wind_velo)

    # traj = trajectory.sin_forward()
    # print(f"================{traj.name}================")
    # contrast_algo(best_p, traj, wind_velo)

    # traj = trajectory.fig8()
    # print(f"================{traj.name}================")
    # contrast_algo(best_p, traj, wind_velo)

    # traj = trajectory.spiral_up()
    # print(f"================{traj.name}================")
    # contrast_algo(best_p, traj, wind_velo)