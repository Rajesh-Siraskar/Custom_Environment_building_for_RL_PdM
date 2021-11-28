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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import signal

class ValveEnv(gym.Env):
    """
    Custom valve environment that follows OpenAI gym interface.
    """
    metadata = {'render.modes': ['console']}
    
    NOISE_MEAN = 0
    NOISE_STD = 1.0
    
    LOW = 0.0
    HIGH = 100.0
    
    def __init__(self, PV_delay=10, K_valve=2.85, Tau_valve=1.00, noise=False):
    
        super(ValveEnv, self).__init__()

        # Set valve parameters
        self.PV_delay = PV_delay
        self.K_valve = K_valve
        self.Tau_valve = Tau_valve
        self.noise = noise
        
        # Initialize the vlave position at 0.0
        self.VP = 0.0
        self.VP_prev = self.VP
        
        # Define action and observation space as gym.spaces objects
        self.action_space = spaces.Box(low=self.LOW, high=self.HIGH, shape=(1,))
        
        # Observation: Error in process variable and input
        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(1,), dtype=np.float32)
        
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        ## Initialize to start randomly at some point
        self.VP = 0.0
        self.VP_prev = self.VP
        return np.array([self.VP]).astype(np.float32)

    def step(self, action_input_signal):
        self.VP = np.exp(-1/self.Tau_valve) * self.VP_prev + self.K_valve * (1 - np.exp(-1/self.Tau_valve)) * action_input_signal

        print('self.VP: ', self.VP)
        self.VP_prev = self.VP

        # Is valve position negative or > HIGH, then done or 
        done = False
        if ((self.VP > self.HIGH) or (self.VP < 0)):
            done = True

        if (self.VP - action_input_signal) < 10:
            reward = 10     
        else:
            reward = 0 

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.VP]).astype(np.float32), reward, done, info


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # # agent is represented as a cross, rest as a dot
        # print("." * self.agent_pos, end="")
        # print("x", end="")
        # print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


PLOT_WIDTH = 16
PLOT_HEIGHT = 4

SIMULATE_FOR_s = 2      # Simulate for N seconds
SAMPLING_FREQ_Hz = 500  # i.e. points per second

INPUT_FF_HIGH = 10      # Input forcing magnitude
INPUT_FREQ_Hz = 5       # Input forcing frequency (i.e. square signal cycles per second)
DUTY_CYCLE = 0.7

PV_DELAY = 10           # Initial delay/inertia
K_valve = 2.85
Tau_valve = 1.00
K_flow = 2.00
Tau_flow = 1.20

NOISE = True            # The process variable PV will have noise
NOISE_MEAN = 0
NOISE_STD = 1.0

# Example: A 5 Hz waveform sampled at 500 Hz for 1 second:
t = np.linspace(0, SIMULATE_FOR_s, SAMPLING_FREQ_Hz, endpoint=False)
u = INPUT_FF_HIGH*signal.square(2*np.pi*(INPUT_FREQ_Hz*t), duty=DUTY_CYCLE)

N = len(u)
VP = np.zeros(N)
PV = np.zeros(N)

for n in range(PV_DELAY, N):
    VP[n] = np.exp(-1/Tau_valve) * VP[n-1] + K_valve * (1 - np.exp(-1/Tau_valve)) * u[n]
    PV[n] = np.exp(-1/Tau_flow) * PV[n-1] + K_flow * (1 - np.exp(-1/Tau_flow)) * VP[n-PV_DELAY]

# Add noise to the process variable
if (NOISE):
    PV = PV + np.random.normal(NOISE_MEAN, NOISE_STD, len(PV))

env = ValveEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env

# Instantiate the env
env = ValveEnv(PV_delay=10, K_valve=2.85, Tau_valve=1.00)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)

#### Train the agent using DQN, PPO or A2C
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(100)

# Test the trained agent
obs = env.reset()
T = 20
for t in range(T):
  action, _ = model.predict(obs, deterministic=True)
  print("Time-step {}".format(t))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break


