import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0
config1 = [sizeLQ, sizeLQ, lightHQ, lightHQ] 
config2 = [sizeHQ, sizeHQ, lightLQ, lightLQ]
config3 = [sizeLQ, sizeLQ, lightLQ, lightLQ]
config4 = [sizeLQ, sizeHQ, lightHQ, lightLQ]
N=10
nDmax=50
theta_base=0.5
rho=600
n=2
max_time=1000
dt = 0.01


times = np.arange(0, max_time, dt)


all_perf_vals = []
all_perf_times = []
configs= [config1, config2, config3, config4] 
labels = ["Low size quality", "Low light quality", "Low size and light quality", "1/2 low size and 1/2 low light quality"]
nD_ranges = [np.arange(1, nDmax)]*3 + [np.arange(2, nDmax, 2)]
for i, config in enumerate(configs):
    perf_val = []
    perf_time = []
    nD_range = nD_ranges[i]
    for nD in nD_range:
        mu = 1 / (1 + nD)  # Probability of finding shelter
        if i != 3:
            s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
        else: 
            s = np.concatenate([[N], np.full(nD // 2, config[0] * N), np.full(nD // 2, config[1] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD // 2, config[2] * theta_base), np.full(nD // 2, config[3] * theta_base)]) # Shelter light levels
        


        # Pack parameters into a dictionary
        params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
        
        # Solve the ODE system
        def model(t, x): return ode_sys(t, x, params)
        x0 = np.zeros(len(s))  # Initial conditions
        sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
        max_val = np.max(sol.y[0])
        max_idx = np.argmax(sol.y[0])

        perf_val.append(max_val/params['N'])

        if max_val/N > 0.5:
            # Find the time at which the solution reaches 95% of its maximum
            time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)
            perf_time.append(time_to_95)
        else:
            perf_time.append(None)

    all_perf_vals.append(perf_val)
    all_perf_times.append(perf_time)


# Set global font size
plt.rcParams.update({'font.size': 22})

colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
fig=plt.figure(figsize = (6,6))
for i in range(len(configs)):
    plt.plot(nD_ranges[i], all_perf_vals[i], linewidth = 4, label = labels[i], c=colors[i])
plt.plot(nD_ranges[0], 1/(nD_ranges[0]), c='black', linewidth = 4, linestyle = '--')
plt.xlabel("Number of distractors")
plt.ylabel("Final proportion under target")
plt.xticks([0, 10, 20, 30, 40, 50])
plt.xlim([0, 50])
# plt.legend()
plt.ylim([-0.1, 1.1])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()





fig=plt.figure(figsize = (6,6))
for i in range(len(configs)):
    plt.plot(nD_ranges[i], all_perf_times[i], linewidth = 4, label = labels[i], c=colors[i])
# plt.scatter(2, all_perf_times[-1][0], marker = '*', s = 200, c='black', zorder=2)
# plt.scatter(10, all_perf_times[-1][4], marker = '^', s = 200, c='black', zorder=2)
# plt.scatter(32, all_perf_times[-1][15], marker = 'P', s = 200, c='black', zorder=2)
plt.xlabel("Number of distractors")
plt.ylabel("$T_{95}$ (s)")
plt.xticks([0, 10, 20, 30, 40, 50])
plt.yticks([0, 200, 400, 600, 800])
plt.ylim([-20, 800])
plt.xlim([0, 50])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.legend()
plt.show()

## ABS FIG

colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
plt.figure(figsize = (6,6))
for i in range(len(configs)):
    if i==2:
        continue
    plt.plot(nD_ranges[i]+1, all_perf_vals[i], linewidth = 4, label = labels[i], c=colors[i])
plt.plot(nD_ranges[0]+1, 1/(nD_ranges[0]+1), c='black', linewidth = 4, linestyle = '--')
plt.xlabel("Number of shelters", size = 18)
plt.ylabel("Final proportion under target", size = 18)
plt.xticks([0, 10, 20, 30, 40, 50], size = 18)
plt.yticks(size = 18)
plt.xlim([0, 50])
# plt.legend()
plt.ylim([-0.1, 1.1])
plt.show()

plt.figure(figsize = (6,6))
for i in range(len(configs)):
    if i==2:
        continue
    plt.plot(nD_ranges[i]+1, all_perf_times[i], linewidth = 4, label = labels[i], c=colors[i])
# plt.scatter(2, all_perf_times[-1][0], marker = '*', s = 200, c='black', zorder=2)
# plt.scatter(10, all_perf_times[-1][4], marker = '^', s = 200, c='black', zorder=2)
# plt.scatter(32, all_perf_times[-1][15], marker = 'P', s = 200, c='black', zorder=2)
plt.xlabel("Number of shelters", size = 18)
plt.ylabel("$T_{95}$ (s)", size = 18)
plt.xticks([0, 10, 20, 30, 40, 50], size = 18)
plt.yticks([0, 200, 400, 600, 800], size = 18)
plt.ylim([-20, 800])
plt.xlim([0, 50])
# plt.legend()
plt.show()


# plt.figure(figsize = (6,4))
# all_perf_vals = []
# all_perf_times = []
# config = [sizeHQ, sizeLQ, lightLQ, lightHQ] 
# labels = ["Low size quality", "3/4 low size + 1/4 low light quality", "1/2 low size + 1/2 low light quality", "1/4 low size + 3/4 low light quality", "Low light quality"]
# colors = ['tab:orange', 'lightsteelblue', 'tab:blue', 'navy', 'tab:red']

# for i, proportion in enumerate(np.arange(5)):
#     perf_val = []
#     perf_time = []
#     if proportion == 0 or proportion == 4:
#         nD_range = np.arange(1, nDmax)
#     elif proportion == 2:
#         nD_range = np.arange(2, nDmax, 2)
#     else:
#         nD_range = np.arange(4, nDmax, 4)
#     for nD in nD_range:
#         mu = 1 / (1 + nD)  # Probability of finding shelter
#         nLightDistractor = nD*proportion//4
#         s = np.concatenate([[N], np.full(nLightDistractor, config[0] * N), np.full(nD-nLightDistractor, config[1] * N)]) # Shelter capacities
#         theta = np.concatenate([[theta_base], np.full(nLightDistractor, config[2] * theta_base), np.full(nD-nLightDistractor, config[3] * theta_base)]) # Shelter light levels
        

#         # Pack parameters into a dictionary
#         params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
        
#         # Solve the ODE system
#         def model(t, x): return ode_sys(t, x, params)
#         x0 = np.zeros(len(s))  # Initial conditions
#         sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)

#         max_val = np.max(sol.y[0])
#         max_idx = np.argmax(sol.y[0])

#         perf_val.append(max_val/params['N'])
#         if max_val/N > 0.5:
#             # Find the time at which the solution reaches 95% of its maximum
#             time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)

#             perf_time.append(time_to_95)
#         else:
#             perf_time.append(np.nan)
#     plt.plot(nD_range, perf_time, label = labels[i], color = colors[i], linewidth = 4)

#     all_perf_vals.append(perf_val)
#     all_perf_times.append(perf_time)

# plt.xlabel("Number of distractors", size = 18)
# plt.ylabel("$T_{95}$ (s)", size = 18)
# # plt.ylabel("Time to 95% completion (s)", size = 18)
# plt.yticks([0, 200, 400, 600, 800], size = 18)
# plt.xticks([0, 10, 20, 30, 40, 50], size = 18)
# plt.ylim([-20, 800])
# # plt.legend()
# plt.show()


