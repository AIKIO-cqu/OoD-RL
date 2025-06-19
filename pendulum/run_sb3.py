import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym


if __name__ == '__main__':
    # environment
    env = gym.make('Pendulum-v1')
    
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model
    log_dir = 'pendulum/tensorboard_logs/sb3'
    model = TD3('MlpPolicy', 
                env, 
                action_noise=action_noise, 
                verbose=1, 
                tensorboard_log=log_dir)
    print(f"模型设备: {model.device}")
    print(f"策略网络设备: {next(model.policy.parameters()).device}")

    # train
    print("==================Training==================")
    model.learn(total_timesteps=125*200,  
                log_interval=1, 
                progress_bar=True)
    print(f"训练完成！")
    print(f"启动TensorBoard命令: tensorboard --logdir {log_dir}")
    env.close()

    # test
    print("==================Testing==================")
    test_env = gym.make('Pendulum-v1', render_mode='human')
    obs, _ = test_env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs, _ = test_env.reset()