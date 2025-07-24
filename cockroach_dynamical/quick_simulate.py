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



distractor_type = "half_size_light"
nD = 6 # Number of distractors

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
max_time=10000
dt = 0.1


times = np.arange(0, max_time, dt)


mu = 1 / (1 + nD)  # Probability of finding shelter
if distractor_type == "half_size_light":
    s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
else:
    s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels



params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}

# Solve the ODE system
def model(t, x): return ode_sys(t, x, params)
x0 = np.zeros(len(s))  # Initial conditions
sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)


max_val = sol.y[0, -1]
max_idx = np.argmax(sol.y[0])

# Find the time at which the solution reaches 95% of its maximum
time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)

# plt.plot(np.arange(0, max_time, dt), sol.y[0]/N, linewidth = 4, label = "Target")
# for i in range(nD//2):
#     plt.plot(np.arange(0, max_time, dt), sol.y[i+1]/N, linewidth = 4, label = "Too bright")
# for i in np.arange(nD//2, nD):
#     plt.plot(np.arange(0, max_time, dt), sol.y[i+1]/N, linewidth = 4, label = "Too large")


# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.xlabel("Time (s)")
# plt.ylabel("Proportion of individuals under shelter")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(np.arange(0, max_time, dt), N-sol.y.sum(axis = 0), linewidth = 4)
# plt.xlabel("Time (s)")
# plt.ylabel("Number of uncommitted cockroaches")
# plt.ylim([-0.5, 9])
# plt.show()



# DYNAMICS FIG
plt.figure(figsize = (6,6))
end_time = 10000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 4, label = "Target", c='black')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 4, label = "Too bright", c='tab:red')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 4, label = "Too large", c='tab:orange')


plt.xlabel("Time (s)", size = 18)
plt.ylabel("Proportion under each shelter", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.ylim([-0.03, 1])
# plt.axhline(0.95*sol.y[0, end_time-1]/N, xmin = 0, xmax = np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()]/1000, linestyle = '--', linewidth = 3, c='black')
# plt.axvline(np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()], ymin = 0, ymax = 0.95*sol.y[0, end_time-1]/N, linestyle = '--', linewidth = 3, c='black')
# plt.legend()
plt.show()

# INSET FIG
# plt.figure(figsize = (8,6))
# end_time = 200
# plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 6, label = "Target")
# plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 6, label = "Too bright")
# plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 6, label = "Too large")

# plt.tick_params(axis='x', direction='in')
# plt.tick_params(axis='y', direction='in')
# plt.xticks([0,10, 20], size = 18)
# plt.yticks([0,0.015, 0.03], size = 18)
# # plt.legend()
# plt.show()

# animate_shelter_numbers(sol.y.T[:500], "cockroach_dynamical_shelter_elimination.mp4", dt = dt)
# decision_time = ((sol.y[0]-sol.y[1])>1).argmax()*dt
# print(decision_time)



plt.figure(figsize = (8,6))
end_time = 10000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time], linewidth = 4, label = "Target")
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time], linewidth = 4, label = "Too bright")
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time], linewidth = 4, label = "Too large")
plt.plot(np.arange(0, end_time*dt, dt), N-sol.y.sum(axis=0)[:end_time], linewidth = 4, alpha = 0.5, linestyle = '--', label = "Uncommitted")
plt.axhline(N/(nD+1), linewidth = 4, c='black')
plt.xlabel("Time (s)", size = 18)
plt.ylabel("Number of cockroaches", size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)

plt.legend()
plt.show()


