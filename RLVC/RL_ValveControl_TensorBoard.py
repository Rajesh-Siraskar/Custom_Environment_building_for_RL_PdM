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
# 
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


import numpy as np
import gym
from gym import spaces
#import stable_baselines3

#from stable_baselines3.common.env_checker import check_env

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


# In[12]:


# class Valve(gym.Env):
#     """
#     Valve Environment that follows gym interface
#     VP: Valve position
#     PV: Process variable e.g. Temperature, flow, pressure etc.
    
#     Action: Input is the Control Signal 
#     Observation: Valve output (valve response) is the VP, PV, reward
#     """
    
#     # Not implemented the GUI ('human' render mode)
#     metadata = {'render.modes': ['console']}
    
#     # Define constants for clearer code
#     NOISE = True
#     NOISE_MEAN = 0.0
#     NOISE_STD = 0.5

#     PV_DELAY = 10

#     K_valve = 2.85
#     Tau_valve = 1.00

#     K_flow = 2.00
#     Tau_flow = 1.20
    
#     def __init__(self, K_valve = 2.85, Tau_valve = 1.00, 
#                  K_flow = 2.00, Tau_flow = 1.20):
        
#         super(Valve, self).__init__()

#         # Define valve model parameters
#         self.K_valve = K_valve
#         self.Tau_valve = Tau_valve

#         self.K_flow = K_flow
#         self.Tau_flow = Tau_flow
    
#         # Initialize the valve position and process variable
#         self.VP = 0.0
#         self.PV = 0.0

#         # Define action and observation space as gym.spaces objects
#         # Continous actions so use Box type
        
#         high = 100.00
#         low = -100.00
#         self.action_space = spaces.Box(low=low, high=high) 
        
#         # The observation will be the valve response (VP and PV)
#         self.observation_space = spaces.Box(low=low, high=high,
#                                             shape=(1,), dtype=np.float32)
        
#     def reset(self):
#         """
#         Important: the observation must be a numpy array
#         :return: (np.array) 
#         """

#         # Initialize the valve position and process variable
#         self.VP = 0.0
#         self.PV = 0.0

#         # here we convert to float32 to make it more general (in case we want to use continuous actions)
#         # return np.array([self.VP, self.PV]).astype(np.float32)
#         return np.array([self.VP, self.PV])

#     def step(self, action):
#         # self.VP = np.exp(-1/self.Tau_valve) * self.VP_prev + K_valve * (1 - np.exp(-1/Tau_valve)) * u[n]
#         # self.PV = np.exp(-1/self.Tau_flow) * PV[n-1] + K_flow * (1 - np.exp(-1/Tau_flow)) * VP[n-PV_DELAY]

#         self.VP = np.exp(-1/self.Tau_valve)
#         self.PV = np.exp(-1/self.Tau_flow)

#         # Are we at the left of the grid?
#         ## if ()
#         ### done = bool(self.agent_pos == 0)

#         reward = 0.0 

#         # Optionally we can pass additional info, we are not using that for now
#         info = {}

#         return (np.array([self.VP, self.PV]), reward, done, info)


#     def render(self, mode='console'):
#         if mode != 'console':
#             raise NotImplementedError()
#         # agent is represented as a cross, rest as a dot
#         print(".", end="")

#     def close(self):
#         pass


# In[13]:


# env = Valve()
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)


# # In[ ]:


# env = Valve(K_valve = 2.85, Tau_valve = 1.00, K_flow = 2.00, Tau_flow = 1.20)

# obs = env.reset()
# env.render()

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())


# In[49]:


#env = Valve(K_valve = 2.85, Tau_valve = 1.00, K_flow = 2.00, Tau_flow = 1.20)

# Instantiate the env
env = GoLeftEnv(grid_size=GRID_SIZE)

# wrap it
#env = make_vec_env(lambda: env, n_envs=1)

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

GO_LEFT = 0
GO_RIGHT = 1
# Hardcoded best agent: always go left!

n_steps = 1000
for step in range(n_steps):
    print("Step {}".format(step + 1))
    
    ## $$$ Rajesh S
    ### ORIGNAL: Always go LEFT
    # obs, reward, done, info = env.step(GO_LEFT)
    ### ORIGNAL
   
    ## Random wind
    random_wind = np.random.randint(0, 100)
    if random_wind < 50:
        obs, reward, done, info = env.step(GO_LEFT)
    else:
        obs, reward, done, info = env.step(GO_RIGHT)
        
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break


# from stable_baselines3 import DQN, PPO, A2C
# from stable_baselines3.common.cmd_util import make_vec_env

# # Instantiate the env
# env = GoLeftEnv(grid_size=GRID_SIZE)
# # wrap it
# env = make_vec_env(lambda: env, n_envs=1)


# # Train the agent
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
# model.learn(10000)

# # Test the trained agent
# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render(mode='console')
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break





