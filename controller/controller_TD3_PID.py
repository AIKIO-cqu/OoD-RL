import numpy as np
from datetime import datetime
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from controller import controller_PID
import gymnasium as gym


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


class PIDEnhancedEnvWrapper(gym.ActionWrapper):
    def __init__(self, env, pid_controller):
        super().__init__(env)
        self.action_low = env.action_low
        self.action_high = env.action_high
        self.pid_controller = pid_controller
    
    def reset(self, **kwargs):
        # 过滤掉不支持的参数
        valid_kwargs = {}
        if hasattr(self.env.reset, '__code__'):
            import inspect
            sig = inspect.signature(self.env.reset)
            for key, value in kwargs.items():
                if key in sig.parameters:
                    valid_kwargs[key] = value
        
        # 如果环境不支持这些参数，就用默认方式调用
        try:
            obs, info = self.env.reset(**valid_kwargs)
        except TypeError:
            # 回退到最基本的调用方式
            obs, info = self.env.reset()

        # 重置PID控制器状态
        self.pid_controller.reset_controller()
        self.pid_controller.reset_time()
        return obs, info
    
    def action(self, action_TD3):
        obs, t, pd, vd, ad, imu, t_last_wind_update = self._extract_info()
        action_PID = self.pid_controller.get_action(obs, t, pd, vd, ad, imu, t_last_wind_update)
        action = action_PID + 0.3 * action_TD3
        action = np.clip(action, self.action_low, self.action_high)  # 限制动作范围
        return action

    def _extract_info(self):
        obs = self.env.get_observation()
        t = self.env.t
        pd,vd,ad = self.env.get_desired(t)
        imu = np.zeros(3)
        t_last_wind_update = self.env.t_last_wind_update
        return obs, t, pd, vd, ad, imu, t_last_wind_update


class TD3Agent():
    def __init__(self, env=None, pid_params=None):
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = f'tensorboard_logs/TD3_PID/{self.time}'
        self.best_model_dir = f'model/TD3_PID/{self.time}'
        self.final_model_dir = f'model/TD3_PID/{self.time}/final_model'

        self.pid_controller = controller_PID.PIDController(pid_params=pid_params)
        self.env = PIDEnhancedEnvWrapper(env, self.pid_controller)

        # 初始化动作噪声
        action_dim = self.env.action_space.shape[0]
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
        self.pid_controller.reset_controller()
    
    def reset_time(self):
        self.pid_controller.reset_time()
    
    def load_model(self, model_path):
        self.model = TD3.load(model_path, env=self.env)
        print(f'TD3 model loaded from: {model_path}')
    
    def save_model(self):
        self.model.save(self.final_model_dir)
        print(f'Final TD3 model saved to: {self.final_model_dir}')
        print(f'Best TD3 model saved to: {self.best_model_dir}')

    def get_action(self, obs, t, pd, vd, ad, imu, t_last_wind_update):
        action_PID = self.pid_controller.get_action(obs, t, pd, vd, ad, imu, t_last_wind_update)
        action_TD3, _ = self.model.predict(obs, deterministic=True)
        action = action_PID + 0.5 * action_TD3
        action = np.clip(action, self.env.action_low, self.env.action_high)  # 限制动作范围
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