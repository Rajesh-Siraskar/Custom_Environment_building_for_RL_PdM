#!/usr/bin/env python
# coding: utf-8

# Valve Modeling
# 
# - Author: Rajesh Siraskar
# - Versions: 
#     - V 2.0: 27-Nov-2021 | Animated valve-flow for creating an environment

#################################

###################################
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import signal

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
    if (NOISE):
        PV[n] = PV[n] + np.random.normal(NOISE_MEAN, NOISE_STD)

x = u
y = VP
z = PV

fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
ax = fig.add_subplot(111)

line,  = ax.plot([],[], 'cadetblue', alpha=0.8, linewidth=3, label='input')
line2, = ax.plot([],[], 'r', alpha=0.5, linewidth=3, label='VP')
line3, = ax.plot([],[], 'b', alpha=0.5, linewidth=3, label='PV')

ax.set_xlim(np.min(t), np.max(t))
ax.set_ylim(np.min(z)*1.2, np.max(z)*1.2)

def animate(i,factor):
    line.set_xdata(t[:i])
    line.set_ydata(u[:i])
    
    line2.set_xdata(t[:i])
    line2.set_ydata(y[:i])
    
    line3.set_xdata(t[:i])
    line3.set_ydata(z[:i])
    
    return line,line2,line3

K=1 # dummy arguments - for example scaling factor
ani = animation.FuncAnimation(fig, animate, frames=len(x), fargs=(K,),
                              interval=10, blit=True)

plt.legend(shadow=True, framealpha=1)
plt.grid(alpha=0.3)
plt.xlabel('t')
plt.title('Valve Flow')
plt.show()

###########################################################
## VALVE STICTION PLOT!!!
###########################################################
# x = u
# y = PV

# fig = plt.figure()
# ax = fig.add_subplot(111)

# line,  = ax.plot([],[], 'b', alpha=0.8, linewidth=0.8, label='input')
# line2, = ax.plot([],[], 'r', alpha=0.5, linewidth=0.8, label='PV')
# ax.set_xlim(np.min(x)*1.2, np.max(x)*1.2)
# ax.set_ylim(np.min(y)*1.2, np.max(y)*1.2)

# def animate_stiction(i,factor):
#     line.set_xdata(x[:i])
#     line.set_ydata(y[:i])
#     line2.set_xdata(x[:i])
#     line2.set_ydata(factor*y[:i])
#     return line,line2

# K = 0.75 # any factor 
# ani = animation.FuncAnimation(fig, animate_stiction, frames=len(x), fargs=(K,),
#                               interval=100, blit=True)

# plt.legend(shadow=True, framealpha=1)
# plt.grid(alpha=0.3)
# plt.xlabel('t')
# plt.title('Valve Flow')
# plt.show()