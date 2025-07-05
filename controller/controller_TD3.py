import numpy as np
from datetime import datetime
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


class DecayNoiseCallback(BaseCallback):
    def __init__(self, verbose=0, decay_rate=0.99, min_noise=0.01):
        super().__init__(verbose)
        self.decay_rate = decay_rate
        self.min_noise = min_noise
        self.current_noise = 0.3
    
    def _on_step(self) -> bool:
        # 使用衰减率递减噪声
        self.current_noise = max(self.current_noise * self.decay_rate, self.min_noise)
        # 更新噪声
        if self.model.action_noise is not None:
            n_actions = self.model.action_noise._sigma.shape[0]
            self.model.action_noise._sigma = self.current_noise * np.ones(n_actions)
        return True


class TD3Agent():
    def __init__(self, env=None):
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = f'tensorboard_logs/TD3/{self.time}'
        self.best_model_dir = f'model/TD3/{self.time}'
        self.final_model_dir = f'model/TD3/{self.time}/final_model'
        self.env = env

        # 初始化动作噪声
        action_dim = env.action_space.shape[0]
        self.action_noise = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=0.3 * np.ones(action_dim)  # 初始噪声水平
        )

        self.model = TD3('MlpPolicy', 
                         self.env, 
                         action_noise=self.action_noise,
                         verbose=1, 
                         tensorboard_log=self.log_dir)
        
    def reset_controller(self):
        pass
    
    def reset_time(self):
        pass
    
    def load_model(self, model_path):
        self.model = TD3.load(model_path, env=self.env)
        print(f'TD3 model loaded from: {model_path}')
    
    def save_model(self):
        self.model.save(self.final_model_dir)
        print(f'Final TD3 model saved to: {self.final_model_dir}')
        print(f'Best TD3 model saved to: {self.best_model_dir}')

    def get_action(self, obs, t, pd, vd, ad, imu, t_last_wind_update):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def train(self, total_timesteps=50*2000, eval_freq=10*2000, n_eval_episodes=1):
        # 如果训练时设置有风，需要直接在环境的reset()方法中定义好风速

        # 创建评估环境
        eval_env = DummyVecEnv([lambda: self.env])

        # 创建保存最佳模型的回调
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.log_dir,
            eval_freq=eval_freq,  # 每多少步评估一次
            n_eval_episodes=n_eval_episodes, # 每次评估多少个episode
            deterministic=True,
            verbose=1
        )

        # 组合回调
        callbacks = [
            DecayNoiseCallback(decay_rate=0.99, min_noise=0.01), 
            eval_callback
        ]

        self.model.learn(total_timesteps=total_timesteps,
                         log_interval=1,
                         progress_bar=True,
                         callback=callbacks)
        self.save_model()
        print("Training completed.")
        print('Start TensorBoard command: tensorboard --logdir tensorboard_logs')