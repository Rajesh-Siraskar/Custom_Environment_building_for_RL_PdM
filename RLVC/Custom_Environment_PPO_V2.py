#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning
# 
# - Author: Rajesh Siraskar
# - Versions: 
#     - V 0.1: 31-Oct-2021 | xxxxxxxx | Initial Stable Baselines working version!
#     - V.0.2: 14-Nov-2021 | 02:06 AM | Initial version! Custom environment
#     - V.0.2: 14-Nov-2021 | 11:48 PM | Add TensorBoard!
#     - V.0.3: 15-Nov-2021 | 13:40 PM | Change reqard function to be more realistic (1-position/GRID_SIZE)
#     - V 0.4: 25-Nov-2021 | 01:32 PM | Python version - Spyder IDE 
# 
# ### Objectives:
# - Create a simple custom environment
# - Train a PPO agent
# - Tensorboard integration:
#     - In code: Simply add the parameter to enable logging ```tensorboard_log="./tensorboard/"```
#     - On Anaconda CLI prompt: 
#         - Activate the conda environment
#         - And it worked in only in ```C:\Users\rajes>``` 
#         - ```C:\Users\rajes>tensorboard --logdir E:\Projects\RL_PdM\tensorboard\.```
# 

# In[46]:


import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

GRID_SIZE = 100

class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left. 
    """
    
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1
    
    def __init__(self, grid_size=20):
        super(GoLeftEnv, self).__init__()

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(1,), dtype=np.float32)
        
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """

        ## $$$ Rajesh S
        ### ORIGNAL
        # Initialize the agent at the right of the grid
        # self.agent_pos = self.grid_size - 1
        ### ORIGNAL

        ## Initialize to start randomly at some point
        self.agent_pos = np.random.randint(low=0, high=self.grid_size)

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)


    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == 0)

        # $$$ original
        # Null reward everywhere except when reaching the goal (left of the grid)
        # reward = 1 if self.agent_pos == 0 else 0

        # $$$ Rajesh 
        # (1 - position/Grid_Size) = reward. At last point this is 1-1 = 0, 
        # Other points as close to left it is higher and higher 
        # everywhere except when reaching the goal (left of the grid)
        if self.agent_pos == 0:
            reward = 10     
        else:
            reward = (1.0 -  self.agent_pos/self.grid_size) 

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass



env = GoLeftEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env

# Instantiate the env
env = GoLeftEnv(grid_size=GRID_SIZE)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)

#### Train the agent using DQN, PPO or A2C
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(10000)

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break





