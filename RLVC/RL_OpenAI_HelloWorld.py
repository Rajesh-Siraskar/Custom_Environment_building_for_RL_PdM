#!/usr/bin/env python
# coding: utf-8

# # OpenAI RL: Stable Baselines Version 3

import gym
from stable_baselines3 import A2C

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)


obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    #print("Info: ", info, " | Reward: ", reward)
    if done:
      obs = env.reset()

print ("Done")
env.render(close=True)


# Model training
from stable_baselines3 import A2C

model = A2C('MlpPolicy', 'CartPole-v1').learn(1000)
print ("Training 1000 episodes done")   


# ### PPO Code
# 
# Also:
# 1. Creating four parallel environments!
# 2. Saving and loading a PPO model


import gym
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# ************ Creating FOUR Parallel environments **************
# env = make_vec_env("CartPole-v1", n_envs=4)
# ************ FOUR Parallel environments **************

env = gym.make('CartPole-v1')

model_PPO = PPO("MlpPolicy", env, verbose=1)
model_PPO.learn(total_timesteps=1000)
model_PPO.save("ppo_cartpole")

del model_PPO # remove to demonstrate saving and loading


model_PPO = PPO.load("ppo_cartpole")
obs = env.reset()

n=0
while n<50:
    action, _states = model_PPO.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    n+=1
    
env.render(close=True)

# model_PPO = PPO.load("ppo_cartpole")
# obs = env.reset()

# for i in range(50):
#     action, _states = model_PPO.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
    
# env.render(close=True)    
