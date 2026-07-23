import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *
import matplotlib.ticker as ticker


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

distractor_type = "low_size"
nD_range = np.arange(2, 14, 2)
max_time=40000
dt = 0.1
uncommitted = np.zeros((len(nD_range), int(max_time/dt)))
solutions = []
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
    theta_base=0.01
    rho=1667
    n=2



    times = np.arange(0, max_time, dt)

    mu = 0.002/(1+nD)
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

    if nD in [2, 6, 12]:
        solutions.append(sol.y[[0, 1]])


lights = np.array([theta_base, theta_base]).reshape(2,1)
sizes = np.array([N, sizeLQ*N]).reshape(2,1)
solutions = np.array(solutions).reshape(6, -1)
two_distractor_leaving_rate = solutions[:2]*lights/(1+params['rho']*(solutions[:2]/sizes)**2)
six_distractor_leaving_rate = solutions[2:4]*lights/(1+params['rho']*(solutions[2:4]/sizes)**2)
twelve_distractor_leaving_rate = solutions[4:]*lights/(1+params['rho']*(solutions[4:]/sizes)**2)

# Set global font size
plt.rcParams.update({'font.size': 24})


fig, axs = plt.subplots(3, 3, figsize = (20,9), sharex=True)


axs[0, 0].plot(np.arange(0, max_time, dt), solutions[0]/N, linewidth = 4, label = "Target", c='black')
axs[0, 0].plot(np.arange(0, max_time, dt), 2*solutions[1]/N, linewidth = 4, label = "Large distractor", c='tab:orange')
axs[0, 0].set_yticks([0, 0.5, 1])

axs[0, 1].plot(np.arange(0, max_time, dt), solutions[2]/N, linewidth = 4, label = "Target", c='black')
axs[0, 1].plot(np.arange(0, max_time, dt), 6*solutions[3]/N, linewidth = 4, label = "Large distractor", c='tab:orange')

axs[0, 2].plot(np.arange(0, max_time, dt), solutions[4]/N, linewidth = 4, label = "Target", c='black')
axs[0, 2].plot(np.arange(0, max_time, dt), 12*solutions[5]/N, linewidth = 4, label = "Large distractor", c='tab:orange')

axs[0, 0].set_ylabel("$Proportion under shelters$")

axs[1, 0].plot(np.arange(0, max_time, dt), two_distractor_leaving_rate[0], linewidth = 4, label = "Target", c='black')
axs[1, 0].plot(np.arange(0, max_time, dt), two_distractor_leaving_rate[1], linewidth = 4, label = "Large distractor", c='tab:orange')
axs[1, 0].set_ylabel("$Q$")
axs[1, 0].set_yticks([0, 0.05, 0.1])
axs[1, 1].plot(np.arange(0, max_time, dt), six_distractor_leaving_rate[0], linewidth = 4, label = "Target", c='black')
axs[1, 1].plot(np.arange(0, max_time, dt), 3*six_distractor_leaving_rate[1], linewidth = 4, label = "Large distractor", c='tab:orange')
axs[1, 2].plot(np.arange(0, max_time, dt), twelve_distractor_leaving_rate[0], linewidth = 4, label = "Target", c='black')
axs[1, 2].plot(np.arange(0, max_time, dt), 6*twelve_distractor_leaving_rate[1], linewidth = 4, label = "Large distractor", c='tab:orange')


axs[2,0].set_ylabel("$Proportion uncommitted$")

axs[2,0].plot(np.arange(0, max_time, dt), uncommitted[0]/N, linewidth = 4, c= 'black')
axs[2,0].set_yticks([0, 0.5, 1.0])
axs[2,1].plot(np.arange(0, max_time, dt), uncommitted[1]/N, linewidth = 4, c= 'black')
# axs[2,2].plot(np.arange(0, max_time, dt), uncommitted[-2], linewidth = 4, c= 'black')
axs[2,2].plot(np.arange(0, max_time, dt), uncommitted[5]/N, linewidth = 4, c= 'black')

hours = np.array([0, 6, 12])
seconds = hours*3600
for i in range(3):
    axs[0, i].set_ylim([-0.1, 1])
    # axs[1, i].set_xlim([-30, 1100])
    axs[1, i].set_ylim([-0.002, 0.1])
    axs[2, i].set_ylim([0, 1])
    axs[2, i].set_xticks(seconds)
    axs[2, i].set_xticklabels(hours)
    axs[2, i].set_xlabel("Time (hrs)")
        
# fig.text(0.5, 0, 'Time (s)', ha='center', va='center', fontsize = 22)
for axis in ['top','bottom','left','right']:
    for i in range(3):
        for j in range(3):
            if j!=0:
                axs[i, j].set_yticklabels([])
            axs[i, j].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.savefig("../figs/uncommitted_analysis.png", bbox_inches = 'tight')
plt.show()


fig = plt.figure(figsize = (8,6))

plt.plot(np.arange(0, max_time, dt), uncommitted[0], linewidth = 4, label = "2 distractors")
plt.plot(np.arange(0, max_time, dt), uncommitted[4], linewidth = 4, label = "10 distractors")
plt.plot(np.arange(0, max_time, dt), uncommitted[-2], linewidth = 4, label = "32 distractors")
plt.xlabel("Time (s)")
plt.ylabel("Number of uncommitted \ncockroaches")
plt.ylim([0, 50])
# plt.xlim([-30, 1150])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()





fig = plt.figure(figsize=(7,5))
skip = 10
im = plt.imshow(uncommitted[:, ::skip], aspect='auto', vmin=0, vmax=60)

plt.yticks(np.arange(len(nD_range))[::2], nD_range[::2])
plt.xlabel("Time (hrs)")
plt.ylabel("Number of distractors")
plt.xticks(seconds, hours)

ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

# cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
# cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')


plt.savefig("../figs/uncommitted_heatmap.png", bbox_inches = 'tight')
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

# time = np.arange(0, max_time, dt)[:50000]
# y = uncommitted[0][:50000]

# # Fit only the initial segment
# tau, A, C = estimate_time_constant(time, y)
# print(f"Estimated time constant (first 50 points): {tau:.2f} s")