import argparse
import numpy as np
import os
import json
from plot import *
import quadsim
import trajectory
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from datetime import datetime, timedelta


def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params


def train(traj, algo_name, log_dir, model_dir):
    # environment
    env = quadsim.Quadrotor(traj=traj, name=algo_name, mode="train")
    
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model
    model = TD3('MlpPolicy', 
                env, 
                # action_noise=action_noise, 
                verbose=1, 
                tensorboard_log=log_dir)
    print(f"模型设备: {model.device}")
    print(f"策略网络设备: {next(model.policy.parameters()).device}")

    eval_callback = EvalCallback(env,
                                 best_model_save_path=model_dir,
                                 log_path=log_dir,
                                 eval_freq=2000,
                                 deterministic=True,
                                 render=False,
                                 n_eval_episodes=1,
                                 verbose=1)
    # train
    print("==================Training==================")
    model.learn(total_timesteps=10*2_000, 
                tb_log_name=f'{algo_name}_{traj.name}', 
                log_interval=1, 
                callback=eval_callback,
                progress_bar=True)
    model.save(f"{model_dir}/final_model")
    print(f"训练完成！")
    print(f"模型已保存到: {model_dir}/final_model")
    print(f"启动TensorBoard命令: tensorboard --logdir {log_dir} --port 6007")


def test(traj, wind_velo, algo_name, model_dir):
    test_rounds = 1
    print(f"==================Testing {algo_name}==================")
    Q = quadsim.Quadrotor(name=algo_name, traj=traj, mode="test")
    ace_error_list = np.empty(test_rounds)
    for round in range(test_rounds):
        seed = 456 + round * 11
        options = {"wind_velo": wind_velo}

        t_readout = -0.0
        p_list = []
        pd_list = []
        
        # model = TD3.load(f"{model_dir}/final_model", env=Q)
        model = TD3.load(f"{model_dir}/best_model", env=Q)
        print(f"模型设备: {model.device}")

        obs, info = Q.reset(seed, options)
        X = obs[0:13]
        t = info['t']
        pbar = tqdm(total=int((Q.params['t_stop']-Q.params['t_start'])/Q.params['dt_readout']) + 1, 
                    desc=f"Round {round+1}/{test_rounds}", unit="rec", leave=True,
                    bar_format='{desc}|{bar}| {n:4d}/{total} [{elapsed}<{remaining}]')
        
        while t < 2.0:  # Q.params['t_stop'] = 20.0
            pd, vd, ad = traj(t)

            # u = np.array([9.81, 0, 0, 0])
            u, _ = model.predict(obs, deterministic=True)
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
        # plot_p(p_list, pd_list)
    ace = np.mean(ace_error_list)
    std = np.std(ace_error_list, ddof=1)
    print(f"*******{algo_name}_{traj.name}*******")
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
    
    # current time
    correct_time = datetime.now()
    time_str = correct_time.strftime("%Y-%m-%d_%H-%M")

    # TensorBoard logs directory
    log_dir = f"tensorboard_logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard日志目录: {log_dir}")
    model_dir = f"model/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    ############## train ##############
    train(traj, algo_name="TD3", log_dir=log_dir, model_dir=model_dir)

    ############## test ##############
    test(traj, wind_velo, algo_name="TD3", model_dir=model_dir)
    # test(traj, wind_velo, algo_name="TD3", model_dir='model/2025-06-19_17-15')