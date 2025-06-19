import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agent.td3 import TD3
import gymnasium as gym
from datetime import datetime

def train(log_dir, model_dir):
    print("==================Training==================")
    agent = TD3(model_dir, s_dim=3, gamma=0.98)
    env = gym.make('Pendulum-v1')

    step = 0
    for episode in range(10):
        s, _ = env.reset()
        for ep_step in range(200):
            a = agent.get_action(s)
            s_, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                episode -= 1
                break
            agent.store_transition(s, a, s_, r)
            s = s_
            step += 1
            if step >= 512:
                break
        if step >= 512:
            break

    for episode in range(200):
        s, _ = env.reset()
        for ep_step in range(200):
            a = agent.get_action(s)
            s_, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
            agent.store_transition(s, a, s_, r)
            s = s_
        print('episode: ', episode, ' reward:', r, ' variance: ', agent.var)
        agent.store_net(str(episode))
        if terminated or truncated:
            episode -= 1
    env.close()

def test(model_dir):
    print("==================Testing==================")
    agent = TD3(model_dir, s_dim=3, gamma=0.98)
    agent.load_net(199)
    env = gym.make('Pendulum-v1', render_mode='human')
    for _ in range(200):
        s, _ = env.reset()
        a = agent.get_action(s)
        s_, r, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            episode -= 1
            break
        s = s_
    env.close()

if __name__ == '__main__':
    # current time
    correct_time = datetime.now()
    time_str = correct_time.strftime("%Y-%m-%d_%H-%M")

    # train
    log_dir = 'pendulum/tensorboard_logs/td3'
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f'pendulum/td3_model/{time_str}'
    os.makedirs(model_dir, exist_ok=True)
    train(log_dir=log_dir, model_dir=model_dir)
    
    # test
    test(model_dir='pendulum\\td3_model\\2025-06-19_23-10')