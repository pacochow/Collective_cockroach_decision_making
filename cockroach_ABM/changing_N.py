import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
import pandas as pd

sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0
config1 = [sizeLQ, sizeLQ, lightHQ, lightHQ] 
config2 = [sizeHQ, sizeHQ, lightLQ, lightLQ]
config3 = [sizeLQ, sizeHQ, lightHQ, lightLQ]
N=100
nDmax=28
theta_base=0.01
rho=1667
n=2
max_time=200000
dt = 0.1


times = np.arange(0, max_time, dt)


all_perf_vals = []
all_perf_times = []
configs= [config1, config2, config3] 
# configs=[config3]
labels = ["Low size quality", "Low light quality", "1/2 low size and 1/2 low light quality"]
nD_range = np.arange(2, nDmax, 2)
for i, config in enumerate(configs):
    perf_val = []
    perf_time = []
    for nD in nD_range:
        mu = 0.002 / (1 + nD)  # Probability of finding shelter
        theta_T = theta_base
        theta_L = theta_base*lightHQ
        theta_B = theta_base*lightLQ
        theta_D = theta_base*config[2]

        S_T = N
        S_L = N*sizeLQ
        S_B = N*sizeHQ
        S_D = N*config[0]
        
        ns = nD+1
        if config!=config3:
            s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
        else: 
            s = np.concatenate([[N], np.full(nD // 2, config[0] * N), np.full(nD // 2, config[1] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD // 2, config[2] * theta_base), np.full(nD // 2, config[3] * theta_base)]) # Shelter light levels
        


        # Pack parameters into a dictionary
        params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
        

        def one_distractor_model(t, xyz):
            X, YD = xyz
            U = N-X-YD
            QT = theta_T / (1.0 + rho * ( (X / S_T)**n ))
            QD = theta_D / (1.0 + rho * ( ((YD / nD) / S_D)**n ))
            dx = -X * QT + mu       * (1.0 - X / S_T)       * U
            dy = -YD * QD + (nD * mu) * (1.0 - YD / (nD * S_D)) * U
            return np.array([dx, dy])


        def two_distractor_model(t, xyz):
            X, YL, YB = xyz
            U = N - X - YL - YB

            # Leaving terms (with density effects)
            QT = theta_T / (1.0 + rho * (X / S_T) ** n)
            QL = theta_L / (1.0 + rho * (2.0 * YL / ((ns - 1) * S_L)) ** n)
            QB = theta_B / (1.0 + rho * (2.0 * YB / ((ns - 1) * S_B)) ** n)

            # Arrival splits: target gets 1/ns, each distractor group gets (ns-1)/(2ns)
            alpha_T  = mu
            alpha_DL = mu * (ns - 1) / 2
            alpha_DB = mu * (ns - 1) / 2

            dX  = -X  * QT + alpha_T  * (1.0 - X  / S_T) * U
            dYL = -YL * QL + alpha_DL * (1.0 - 2.0 * YL / ((ns - 1) * S_L)) * U
            dYB = -YB * QB + alpha_DB * (1.0 - 2.0 * YB / ((ns - 1) * S_B)) * U
            return np.array([dX, dYL, dYB])

        times = np.arange(0.0, max_time, dt)
        
        
        if config!=config3:
            x0 = np.array([0.0, 0.0])   # start uncommitted
            sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
        else:
            x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
            sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

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



model = "uncommitted_capacity"
configs = ["low_size", "low_light", "half_size_light"]
# configs = ["half_size_light"]
labels = ["Low size quality", "Low light quality", "1/2 low size and 1/2 low light quality"]
colors = ['tab:orange', 'tab:red', 'dodgerblue']
nDmax = 28
# agent_counts = [10, 100, 1000]
agent_counts = [15, 25, 50]

# Set global font size
plt.rcParams.update({'font.size': 24})


fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
for ax, N in zip(axes.flat, agent_counts):
    for i, config in enumerate(configs):
        data_path = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_500decay_target_proportions.npy"
        target_proportions = np.load(data_path)[:13]
        median_proportions = np.nanmedian(target_proportions, axis=1)
        nD_range = np.arange(2, nDmax, 2)
        
        ax.plot(nD_range, median_proportions, c=colors[i], linewidth = 3)
        ax.scatter(nD_range, median_proportions, c=colors[i], marker='X', alpha = 0.5)
        ax.plot(nD_range, all_perf_vals[i], c= colors[i], linestyle = '--', linewidth = 3)
    ax.set_title(f"{N} agents")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Number of distractors")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
axes[0].set_ylabel("Final proportion under target")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
for ax, N in zip(axes.flat, agent_counts):
    for i, config in enumerate(configs):
        data_path = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_500decay_time_constants.npy"
        time_constants = np.load(data_path)[:13]
        median_time_constants = np.nanmean(time_constants, axis=1)
        nD_range = np.arange(2, nDmax, 2) 
        ax.plot(nD_range, median_time_constants, c=colors[i], linewidth = 3)
        ax.scatter(nD_range, median_time_constants, c=colors[i], marker='X', alpha = 0.5)
        ax.plot(nD_range, all_perf_times[i], c=colors[i], linestyle = '--', linewidth = 3)
    ax.set_title(f"{N} agents")
    ax.set_xticks([0, 20])
    ax.set_xlabel("Number of distractors")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
axes[0].set_ylabel("$T_{95}$ (s)")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()




fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True)
hours = np.array([0, 12, 24, 36])
seconds = hours*3600
for col_idx, N in enumerate(agent_counts):
    for i, config in enumerate(configs):
        color = colors[i]
        label = labels[i]
        nD_range = np.arange(2, nDmax, 2) 

        # Top row: Final target proportion
        data_path_val = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_500decay_target_proportions.npy"
        target_proportions = np.load(data_path_val)[:13]
        median_proportions = np.nanmedian(target_proportions, axis=1)
        axes[0, col_idx].plot(nD_range, median_proportions, c=color, linewidth=3)
        axes[0, col_idx].scatter(nD_range, median_proportions, c=color, marker='X', alpha=0.5)
        axes[0, col_idx].plot(nD_range, all_perf_vals[i], c=color, linestyle='--', linewidth=3)
        axes[0, col_idx].set_title(f"{N} agents")
        axes[0, col_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            
        # Bottom row: T95
        data_path_time = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_500decay_time_constants.npy"
        time_constants = np.load(data_path_time)[:13]
        counts = np.sum(~np.isnan(time_constants), axis=1)
        median_time_constants = np.nanmedian(time_constants,axis = 1)
        median_time_constants[counts<=10] = np.nan

        axes[1, col_idx].plot(nD_range, median_time_constants, c=color, linewidth=3)
        axes[1, col_idx].scatter(nD_range, median_time_constants, c=color, marker='X', alpha=0.5)
        axes[1, col_idx].plot(nD_range, all_perf_times[i], c=color, linestyle='--', linewidth=3)
        # axes[1, col_idx].set_yticks([0, 50000, 100000, 150000, 200000])
        axes[1, col_idx].set_xlabel("Number of distractors")
        # axes[1, col_idx].set_ylim([0, 90000])
    # Common formatting
    for row in range(2):
        axes[row, col_idx].set_xticks([0, 10, 20])
        
        axes[1, col_idx].set_yticks(seconds)
        axes[1, col_idx].set_yticklabels(hours)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[row, col_idx].spines[axis].set_linewidth(2)

axes[0, 0].set_ylabel("Final proportion under target")
axes[1, 0].set_ylabel("$T_{95}$ (hrs)")



plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
