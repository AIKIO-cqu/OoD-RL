import numpy as np
import matplotlib.pyplot as plt

def plot_p(p_list, pd_list):
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
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    axes[0].plot(motor_speed_dot[:, 0], label='Motor Speed 1 Dot', color='blue')
    axes[1].plot(motor_speed_dot[:, 1], label='Motor Speed 2 Dot', color='orange')
    axes[2].plot(motor_speed_dot[:, 2], label='Motor Speed 3 Dot', color='green')
    axes[3].plot(motor_speed_dot[:, 3], label='Motor Speed 4 Dot', color='red')
    axes[0].set_title('Motor Speed Dots')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

def plot_r(reward_list):
    reward_list = np.array(reward_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(reward_list, label='Reward per Step', color='blue')
    ax.set_title('Reward per Step Over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.grid()
    plt.show()

def plot_err(pos_error_list, yaw_error_list, vel_error_list):
    pos_error_list = np.array(pos_error_list)
    yaw_error_list = np.array(yaw_error_list)
    vel_error_list = np.array(vel_error_list)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    axes[0].plot(pos_error_list, label='Position Error', color='blue')
    axes[1].plot(yaw_error_list, label='Yaw Error', color='orange')
    axes[2].plot(vel_error_list, label='Velocity Error', color='green')
    axes[0].set_title('Errors Over Time')
    plt.xlabel('Time Step')
    for ax in axes:
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_action(action_list):
    action_list = np.array(action_list)
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    axes[0].plot(action_list[:, 0], label='a_z', color='red')
    axes[1].plot(action_list[:, 1], label='omega_x', color='orange')
    axes[2].plot(action_list[:, 2], label='omega_y', color='green')
    axes[3].plot(action_list[:, 3], label='omega_z', color='blue')
    axes[0].set_title('Actions Over Time')
    plt.xlabel('Time Step')
    for ax in axes:
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()