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
        sol
    Returns:
        dx (array): Rate of change of the system state.
    """
    D = x / p['s']  # Compute densities
    Q = p['theta'] / (1 + p['rho'] * D**p['n'])  # Light-influence function
    dx = -x * Q + p['mu'] * (1 - D) * (p['N'] - np.sum(x))  # Rate of change
    return dx



distractor_type = "half_size_light"
nD = 8 # Number of distractors

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
theta_base=0.01
rho=1667
n=2
max_time=1000000
dt = 0.1


times = np.arange(0, max_time, dt)

mu = 0.002/(1+nD)
if distractor_type == "half_size_light":
    # s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
    # theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels

    s = np.concatenate([[N], np.full(6, config[0] * N), np.full(2, config[1] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(6, config[2] * theta_base), np.full(2, config[3] * theta_base)]) # Shelter light levels
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



print(f"Decision time: {time_to_95}")

hours = np.array([0, 12, 24])
seconds = hours*3600
# DYNAMICS FIG
# Set global font size
plt.rcParams.update({'font.size': 24})

fig = plt.figure(figsize = (4,6))
end_time = 1000000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 4, label = "Target", c='black')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 4, label = "Too bright", c='tab:red')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 4, label = "Too large", c='tab:orange', linestyle = '--')
plt.xlabel("Time (hrs)")
# plt.ylabel("Proportion under \neach shelter")
# plt.ylabel("$Z_i$")
plt.xticks(seconds, hours)
plt.ylim([-0.03, 1])
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.axhline(0.95*sol.y[0, end_time-1]/N, xmin = 0, xmax = np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()]/1000, linestyle = '--', linewidth = 3, c='black')
# plt.axvline(np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()], ymin = 0, ymax = 0.95*sol.y[0, end_time-1]/N, linestyle = '--', linewidth = 3, c='black')
# plt.savefig(f"../figs/mfm_{nD}_simulation_dynamics.png", bbox_inches = 'tight')
plt.show()


### INSET FIG 2E
fig = plt.figure(figsize = (2,2))
end_time = 60000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 4, label = "Target", c='black')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 4, label = "Too bright", c='tab:red')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 4, label = "Too large", c='tab:orange')
plt.scatter(np.arange(0, end_time*dt, dt)[2000], sol.y[0, 2000]/N, marker = 's', s=100, zorder = 4, c='yellow', edgecolors = 'black')
plt.scatter(np.arange(0, end_time*dt, dt)[20000], sol.y[0, 20000]/N, marker = 'X', s=100, zorder = 4, c='yellow', edgecolors = 'black')
plt.xticks([], [])
plt.yticks([], [])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()

plt.rcParams.update({'font.size': 16})
### INSET FIG 3A
hours = np.array([0, 4])
seconds = hours*3600
fig = plt.figure(figsize = (1,1))
end_time = 180000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 4, label = "Target", c='black')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 4, label = "Too bright", c='tab:red')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 4, label = "Too large", c='tab:orange')
plt.xticks(seconds, hours)
# plt.xticks([], [])
plt.yticks([0, 0.05])
plt.ylim([0, 0.05])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()



## FIG 3A
plt.rcParams.update({'font.size': 24})
# fig = plt.figure(figsize = (4,6))
fig = plt.figure(figsize = (2.5,4.25))
end_time = 500000
# end_time = 1000000
plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/N, linewidth = 4, label = "Target", c='black')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[1, :end_time]/N, linewidth = 4, label = "Too bright", c='tab:red')
plt.plot(np.arange(0, end_time*dt, dt), sol.y[-1, :end_time]/N, linewidth = 4, label = "Too large", c='tab:orange')
plt.scatter(np.arange(0, end_time*dt, dt)[2000], sol.y[0, 2000]/N, marker = 's', s=150, zorder = 4, c='yellow', edgecolors = 'black')
plt.scatter(np.arange(0, end_time*dt, dt)[20000], sol.y[0, 20000]/N, marker = 'X', s=150, zorder = 4, c='yellow', edgecolors = 'black')
# plt.scatter(np.arange(0, end_time*dt, dt)[120000], sol.y[0, 120000]/N, marker = 'o', s=150, zorder = 4, c='yellow', edgecolors = 'black')
# plt.scatter(np.arange(0, end_time*dt, dt)[500000], sol.y[0, 500000]/N, marker = '*', s=150, zorder = 4, c='yellow', edgecolors = 'black')
plt.xlabel("Time (hrs)")
# plt.xticks([0, 500, 1000])
hours = np.array([0, 6, 12])
seconds = hours*3600
plt.xticks(seconds, hours)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.5, 1])
# plt.ylabel("Proportion under each shelter")
plt.ylabel("$X$")
plt.ylim([-0.03, 1])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.axhline(0.95*sol.y[0, end_time-1]/N, xmin = 0, xmax = np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()]/1000, linestyle = '--', linewidth = 3, c='black')
# plt.axvline(np.arange(0, end_time*dt, dt)[np.abs(sol.y[0, :end_time]/N - 0.95*sol.y[0, end_time-1]/N).argmin()], ymin = 0, ymax = 0.95*sol.y[0, end_time-1]/N, linestyle = '--', linewidth = 3, c='black')
# plt.savefig("../figs/conjunction_simulate_with_markers.png", bbox_inches = 'tight')
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




# fig = plt.figure(figsize = (5,6))
# end_time = max_time
# plt.plot(np.arange(0, end_time*dt, dt), sol.y[0, :end_time]/(sol.y[1, :end_time]+0.000001), linewidth = 4, label = "Target", c='black')
# plt.xlabel("Time (s)")
# plt.ylabel(r"$\frac{X_1}{X_2}$")

# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
# plt.ylim([0, 250])
# plt.show()
