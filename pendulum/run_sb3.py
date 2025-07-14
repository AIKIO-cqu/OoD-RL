import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym


class ActionScaleWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        return action * 2.0

class TD3Agent():
    def __init__(self, env):
        self.env = env
        self.log_dir = 'pendulum/tensorboard_logs/sb3'
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        self.model = TD3('MlpPolicy', 
                         env, 
                         action_noise=action_noise, 
                         verbose=1, 
                         tensorboard_log=self.log_dir)
    
    def train(self, total_timesteps=150*200, log_interval=1):
        self.model.learn(total_timesteps=total_timesteps, 
                         log_interval=log_interval, 
                         progress_bar=True)
        print("Training completed!")
        print(f"Start TensorBoard command: tensorboard --logdir {self.log_dir}")
    
    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


if __name__ == '__main__':
    # environment
    env = gym.make('Pendulum-v1')
    env = ActionScaleWrapper(env)

    # agent
    agent = TD3Agent(env)
    agent.train(total_timesteps=150*200, log_interval=1)

    # test
    test_env = gym.make('Pendulum-v1', render_mode='human')
    obs, _ = test_env.reset()
    for _ in range(200):
        action = agent.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs, _ = test_env.reset()