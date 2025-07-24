import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *


# Function to define the ODE system
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

distractor_type = "half_size_light"
nD_range = np.arange(2, 36, 2)
max_time=1000
dt = 0.1
uncommitted = np.zeros((len(nD_range), int(max_time/dt)))
for i, nD in enumerate(nD_range):


    sizeLQ = 1.75
    sizeHQ = 1.0
    lightLQ = 1.75
    lightHQ = 1.0
    configs = {
        "low_size": [sizeLQ, sizeLQ, lightHQ, lightHQ],
        "low_light": [sizeHQ, sizeHQ, lightLQ, lightLQ],
        "low_size_light": [sizeLQ, sizeLQ, lightLQ, lightLQ],
        "half_size_light": [sizeHQ, sizeLQ, lightLQ, lightHQ],
    }
    config = configs[distractor_type]
    N=100
    theta_base=0.5
    rho=600
    n=2



    times = np.arange(0, max_time, dt)


    mu = 1 / (1 + nD)  # Probability of finding shelter
    if distractor_type == "half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels



    # Pack parameters into a dictionary
    params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}

    # Solve the ODE system
    def model(t, x): return ode_sys(t, x, params)
    x0 = np.zeros(len(s))  # Initial conditions
    sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)

    uncommitted[i] = N-sol.y.sum(axis=0)


# Set global font size
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize = (8,6))

plt.plot(np.arange(0, max_time, dt), uncommitted[0], linewidth = 4, label = "2 distractors")
plt.plot(np.arange(0, max_time, dt), uncommitted[4], linewidth = 4, label = "10 distractors")
plt.plot(np.arange(0, max_time, dt), uncommitted[-2], linewidth = 4, label = "32 distractors")
plt.xlabel("Time (s)")
plt.ylabel("Number of uncommitted \ncockroaches")
plt.ylim([0, 50])
plt.xlim([-30, 1150])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()

fig=plt.figure(figsize = (8,6))
skip = 10
plt.imshow(uncommitted[:, ::skip], aspect='auto', vmin = 0, vmax = 30)
plt.yticks(np.arange(len(nD_range))[::2], nD_range[::2])
plt.xticks(np.array([0, 2000, 4000, 6000, 8000, 10000])/skip, [0, 200, 400, 600, 800, 1000], size = 18)
plt.xlabel("Time (s)")
plt.ylabel("Number of distractors")
cbar = plt.colorbar()
cbar.set_label("Number of uncommitted \ncockroaches")
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()







# # MEASURE TIME CONSTANT OF UNCOMMITTED DECAY

# def estimate_time_constant(time, y, plot=True):
#     # Exponential decay model with asymptote
#     def exp_decay(t, A, tau, C):
#         return A * np.exp(-t / tau) + C

#     # Initial guesses: A = starting y - ending y, tau = ~1/5 of total time, C = final y
#     A0 = y[0] - y[-1]
#     tau0 = (time[-1] - time[0]) / 5
#     C0 = y[-1]

#     popt, _ = curve_fit(exp_decay, time, y, p0=(A0, tau0, C0))
#     A_fit, tau_fit, C_fit = popt
#     y_fit = exp_decay(time, *popt)

#     if plot:
#         plt.figure(figsize=(10, 6))
#         plt.plot(time, y, label="Data", linewidth=2)
#         plt.plot(time, y_fit, '--', label=f"Fit: $\\tau$ ≈ {tau_fit:.1f}", linewidth=3)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Value")
#         plt.title("Exponential Decay with Asymptote")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     return tau_fit, A_fit, C_fit

# time = np.arange(0, max_time, dt)[:50]
# y = uncommitted[0][:50]

# # Fit only the initial segment
# tau, A, C = estimate_time_constant(time, y)
# print(f"Estimated time constant (first 50 points): {tau:.2f} s")