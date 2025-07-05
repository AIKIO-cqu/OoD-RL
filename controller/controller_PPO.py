import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class PPOAgent():
    def __init__(self, env=None):
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = f'tensorboard_logs/PPO/{self.time}'
        self.best_model_dir = f'model/PPO/{self.time}'
        self.final_model_dir = f'model/PPO/{self.time}/final_model'
        self.env = env

        self.model = PPO('MlpPolicy', 
                         self.env, 
                         verbose=1, 
                         tensorboard_log=self.log_dir)
        
    def reset_controller(self):
        pass
    
    def reset_time(self):
        pass
    
    def load_model(self, model_path):
        self.model = PPO.load(model_path, env=self.env)
        print(f'PPO model loaded from: {model_path}')
    
    def save_model(self):
        self.model.save(self.final_model_dir)
        print(f'Final PPO model saved to: {self.final_model_dir}')
        print(f'Best PPO model saved to: {self.best_model_dir}')

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

        self.model.learn(total_timesteps=total_timesteps,
                         log_interval=1,
                         progress_bar=True,
                         callback=eval_callback)
        self.save_model()
        print("Training completed.")
        print('Start TensorBoard command: tensorboard --logdir tensorboard_logs')