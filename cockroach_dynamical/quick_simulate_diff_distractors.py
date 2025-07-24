import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *

def ode_sys(t, x, p):
    """
    Compute the rate of change for the ODE system.
    
    Parameters:
        t (float): Time.
        x (array): Current state of the system (individuals in each shelter).
        p (dict): Dictionary containing parameters (s, theta, mu, rho, n, N).
        
    Returns:
        dx (array): Rate of change of the system state.
    """
    D = x / p['s']  # Compute densities
    Q = p['theta'] / (1 + p['rho'] * D**p['n'])  # Light-influence function
    dx = -x * Q + p['mu'] * (1 - D) * (p['N'] - np.sum(x))  # Rate of change
    return dx



nD = 6 # Number of distractors

N=100
theta_base=0.5
rho=600
n=2
max_time=10000
dt = 0.1


times = np.arange(0, max_time, dt)


mu = 1 / (1 + nD)  # Probability of finding shelter

s = np.array([100, 120, 125, 130, 100, 100, 100])
theta = np.array([0.5, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9])

# Pack parameters into a dictionary
params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}

# Solve the ODE system
def model(t, x): return ode_sys(t, x, params)
x0 = np.zeros(len(s))  # Initial conditions
sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)


max_val = sol.y[0, -1]
max_idx = np.argmax(sol.y[0])

# Find the time at which the solution reaches 95% of its maximum
time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)


# DYNAMICS FIG
plt.figure(figsize = (6,6))
end_time = 5000
for i in range(7):
    if i == 0:
        plt.plot(np.arange(0, end_time*dt, dt), sol.y[i, :end_time]/N, linewidth = 4, label = f"Target", c='black')
    elif i < (nD/2)+1:
        plt.plot(np.arange(0, end_time*dt, dt), sol.y[i, :end_time]/N, linewidth = 4, label = f"Large distractor {i}", c='tab:red', alpha = 1-i/10)
    else:
        plt.plot(np.arange(0, end_time*dt, dt), sol.y[i, :end_time]/N, linewidth = 4, label = f"Bright distractor {i-3}", c='tab:orange', alpha = 1-i/10)
plt.xlabel("Time (s)", size = 18)
plt.ylabel("Proportion under each shelter", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.ylim([-0.03, 1])
# plt.axhline(0.95*sol.y[0, end_time-1]/N, xmin = 0, xmax = np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()]/1000, linestyle = '--', linewidth = 3, c='black')
# plt.axvline(np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()], ymin = 0, ymax = 0.95*sol.y[0, end_time-1]/N, linestyle = '--', linewidth = 3, c='black')
plt.legend()
plt.show()