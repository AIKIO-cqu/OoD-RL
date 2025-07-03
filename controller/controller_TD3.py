from datetime import datetime
from stable_baselines3 import TD3


class TD3Agnet():
    def __init__(self):
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = f'tensorboard_logs/{self.time}'
        self.model_dir = f'model/td3/{self.time}'
    
    def set_env(self, env):
        self.env = env
    
    def reset_controller(self):
        self.model = TD3('MlpPolicy', 
                         self.env, 
                         verbose=1, 
                         tensorboard_log=self.log_dir)
    
    def reset_time(self):
        pass

    def get_action(self, obs, t, pd, vd, ad, imu, t_last_wind_update):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps,
                         log_interval=1,
                         progress_bar=True)
        self.model.save(self.model_dir)
        print(f'TD3 model saved to: {self.model_dir}')
        print('Start TensorBoard command: tensorboard --logdir')