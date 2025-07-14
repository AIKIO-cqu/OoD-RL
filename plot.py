import matplotlib.pyplot as plt
import numpy as np

def plot_traj(p_list, pd_list, algo_name, time=None):
    p_list = np.array(p_list)
    pd_list = np.array(pd_list)
    # 绘制3D轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p_list[:, 0], p_list[:, 1], p_list[:, 2], label='Actual Trajectory', color='blue')
    ax.plot(pd_list[:, 0], pd_list[:, 1], pd_list[:, 2], label='Desired Trajectory', color='orange', linestyle='--')
    ax.scatter(p_list[0, 0], p_list[0, 1], p_list[0, 2], color='red', s=100, label='Start Point')
    ax.scatter(p_list[-1, 0], p_list[-1, 1], p_list[-1, 2], color='green', s=100, label='End Point')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Quadrotor Trajectory ({algo_name})')
    ax.legend()
    if time is not None:
        plt.savefig(f'figure/{algo_name}/{time}.png', dpi=300)
    plt.show()

def plot_reward(reward_list, algo_name, time=None):
    reward_list = np.array(reward_list)
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(reward_list, label='Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(f'Reward Curve ({algo_name})')
    plt.legend()
    plt.grid()
    if time is not None:
        plt.savefig(f'figure/{algo_name}/{time}_reward.png', dpi=300)
    plt.show()