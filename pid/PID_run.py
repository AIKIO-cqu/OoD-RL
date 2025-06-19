import argparse
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import *
from PID_quadsim import Quadrotor_PID
import trajectory
from tqdm import tqdm
import PID_controller


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params


def test_pid(traj, wind_velo, algo_name):
    print("==================test PID==================")
    best_p = readparamfile("params/pid.json")
    C = PID_controller.PIDController(given_pid=True, Lam_xy=best_p['Lam_xy'], K_xy=best_p['K_xy'], 
                                    Lam_z=best_p['Lam_z'], K_z=best_p['K_z'], i=best_p['i'])
    Q = Quadrotor_PID(pid_controller=C, name=algo_name, traj=traj, mode="test")

    print("Testing " + algo_name)
    ace_error_list = np.empty(10)
    for round in range(10):
        seed = 456 + round * 11
        options = {"wind_velo": wind_velo}

        t_readout = -0.0
        p_list = []
        pd_list = []
        action_list = []

        obs, info = Q.reset(seed, options)
        X = obs[0:13]
        t = info['t']
        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']) + 1, 
                    desc=f"Round {round+1}/10", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        while t < 2.0:  # Q.params['t_stop'] = 20.0
            pd, vd, ad = traj(t)

            action, u = Q.pid_controller.getu(Q.X, Q.t, pd, vd, ad, Q.imu, Q.t_last_wind_update)
            action_list.append(action)
            obs, reward, terminated, truncated, info = Q.step(u)
            X = obs[0:13]
            t = info['t']
            
            if t >= t_readout:
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

        plot_p(p_list, pd_list)
        plot_r(Q.reward_list)
        plot_err(Q.pos_error_list, Q.yaw_error_list, Q.vel_error_list)
        plot_action(action_list)
        
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
    
    ############## test ##############
    test_pid(traj, wind_velo, algo_name="PID")