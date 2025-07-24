import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp


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



model = "uncommitted_capacity"
configs = ["low_size", "low_light", "low_size_light", "half_size_light"]
labels = ["Low size quality", "Low light quality", "Low size and light quality", "1/2 low size and 1/2 low light quality"]
colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
nDmax = 50
agent_counts = [10, 30, 50, 100, 200, 1000]

# Set global font size
plt.rcParams.update({'font.size': 22})


fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
for ax, N in zip(axes.flat, agent_counts):
    for i, config in enumerate(configs):
        data_path = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_1decay_target_proportions.npy"
        target_proportions = np.load(data_path)
        mean_proportions = np.nanmedian(target_proportions, axis=1)
        nD_range = np.arange(2, nDmax, 2) if config == "half_size_light" else np.arange(1, nDmax)
        
        ax.plot(nD_range, mean_proportions, c=colors[i], linewidth = 3)
        ax.scatter(nD_range, mean_proportions, c=colors[i], marker='X', alpha = 0.5)
        ax.plot(nD_range, all_perf_vals[i], c= colors[i], linestyle = '--', linewidth = 3)
    ax.set_title(f"{N} agents")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Number of distractors")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
axes[0, 0].set_ylabel("Final proportion under target")
axes[1, 0].set_ylabel("Final proportion under target")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
for ax, N in zip(axes.flat, agent_counts):
    for i, config in enumerate(configs):
        data_path = f"../Analysed_data/cockroach_abm_{model}/{N}_{config}_1decay_time_constants.npy"
        time_constants = np.load(data_path)
        mean_time_constants = np.nanmean(time_constants, axis=1)
        nD_range = np.arange(2, nDmax, 2) if config == "half_size_light" else np.arange(1, nDmax)
        ax.plot(nD_range, mean_time_constants, c=colors[i], linewidth = 3)
        ax.scatter(nD_range, mean_time_constants, c=colors[i], marker='X', alpha = 0.5)
        ax.plot(nD_range, all_perf_times, c=colors[i], linestyle = '--', linewidth = 3)
    ax.set_title(f"{N} agents")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim([0, 1000])
    ax.set_xlabel("Number of distractors")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
axes[0, 0].set_ylabel("$T_{95}$ (s)")
axes[1, 0].set_ylabel("$T_{95}$ (s)")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
