import argparse
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import quadsim
import trajectory
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
import controller_PID
from datetime import datetime, timedelta


# global variables to record the results
ACE_DICT = {}
STD_DICT = {}

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

def test(traj, Wind_velo, Name, time_str):
    global ACE_DICT, STD_DICT

    # best_p = readparamfile('OoD-RL/params/pid.json')
    # C = controller_PID.PIDController(given_pid=True, Lam_xy=best_p['Lam_xy'], K_xy=best_p['K_xy'], 
    #                                  Lam_z=best_p['Lam_z'], K_z=best_p['K_z'], i=best_p['i'])
    # Q = quadsim.Quadrotor(pid_controller=C, name=Name, traj=traj, state="test")
    Q = quadsim.Quadrotor(name=Name, traj=traj, state="test")
    check_env(Q)

    print("Testing " + Name)
    ace_error_list = np.empty(10)
    for round in range(10):
        seed = 456 + round * 11
        options = {"wind_velo": Wind_velo}

        t_readout = -0.0
        p_list = []
        pd_list = []
        Z_list = []
        
        model = TD3.load(f"OoD-RL/model/TD3_{time_str}")
        print(f"模型设备: {model.device}")

        obs, info = Q.reset(seed, options)
        X = obs[0:13]
        t = info['t']
        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']) + 1, 
                    desc=f"Round {round+1}/10", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        while t < Q.params['t_stop']:
            pd, vd, ad = traj(t)

            # u = Q.pid_controller.getu(Q.X, Q.t, pd, vd, ad, Q.imu, Q.t_last_wind_update)
            u, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = Q.step(u)
            X = obs[0:13]
            t = info['t']
            Z_list.append(info['Z'])
            
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
        # plot_(p_list, pd_list)
        plot_Z(Z_list)
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    ACE_DICT[Name] = ace
    STD_DICT[Name] = std
    print("*******", Name, "*******")
    print("ACE Error: %.3f(%.3f)" % (ace, std))
    return np.mean(ace_error_list)

def plot_(p_list, pd_list):
    p_list = np.array(p_list)
    pd_list = np.array(pd_list)
    print("p_list.shape:", p_list.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_list[:, 0], p_list[:, 1], p_list[:, 2], label='track')
    ax.scatter(pd_list[:, 0], pd_list[:, 1], pd_list[:, 2], c='r', marker='o', label='target')
    ax.view_init(azim=45., elev=30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def plot_Z(Z_list):
    Z_list = np.array(Z_list)
    motor_speed_dot = []
    for i in range(Z_list.shape[0]-1):
        motor_speed_dot.append(Z_list[i+1, :] - Z_list[i, :])
    motor_speed_dot = np.array(motor_speed_dot)
    # print("motor_speed_dot.shape:", motor_speed_dot.shape)
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    axes[0].plot(motor_speed_dot[:, 0], label='Motor Speed 1 Dot', color='blue')
    axes[1].plot(motor_speed_dot[:, 1], label='Motor Speed 2 Dot', color='orange')
    axes[2].plot(motor_speed_dot[:, 2], label='Motor Speed 3 Dot', color='green')
    axes[3].plot(motor_speed_dot[:, 3], label='Motor Speed 4 Dot', color='red')
    axes[0].set_title('Motor Speed Dots')
    plt.xlabel('Time Step')
    plt.savefig('motor_speed_dot_analysis.png')

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

    # current time
    correct_time = datetime.now() + timedelta(hours=8)
    time_str = correct_time.strftime("%m-%d_%H-%M")

    # TensorBoard logs directory
    log_dir = f"./OoD-RL/tensorboard_logs/TD3_{time_str}/"
    os.makedirs(log_dir, exist_ok=True)

    # create the environment and model
    env = quadsim.Quadrotor(traj=traj, name="RL", state="train")
    check_env(env)
    model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    print(f"模型设备: {model.device}")
    print(f"策略网络设备: {next(model.policy.parameters()).device}")
    print(f"TensorBoard日志目录: {log_dir}")

    # train
    print("==================train==================")
    model.learn(total_timesteps=100*5_000, progress_bar=True)
    model.save(f"OoD-RL/model/TD3_{time_str}")
    print(f"训练完成！")
    print(f"模型已保存到: OoD-RL/model/TD3_{time_str}")
    print(f"启动TensorBoard命令: tensorboard --logdir {log_dir} --port 6007")

    # # test
    # print("==================test==================")
    # print("Current time:", time_str)
    # test(traj, Wind_velo, "RL", time_str=time_str)