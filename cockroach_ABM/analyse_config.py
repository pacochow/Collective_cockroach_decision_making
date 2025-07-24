import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
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


distractor_type = "half_size_light"
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
nDmax=50
theta_base=0.5
rho=600
n=2
max_time=2000
dt = 0.01

if distractor_type=="half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)


times = np.arange(0, max_time, dt)

perf_val = []
perf_time = []
for nD in nD_range:
    mu = 1 / (1 + nD)  # Probability of finding shelter

    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    

    # Pack parameters into a dictionary
    params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
    
    # Solve the ODE system
    def model(t, x): return ode_sys(t, x, params)
    x0 = np.ones(len(s))*N/len(s)  # Initial conditions
    sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
    max_val = sol.y[0, -1]
    max_idx = np.argmax(sol.y[0])

    perf_val.append(max_val/params['N'])
    if max_val/N > 0.5:
        # Find the time at which the solution reaches 95% of its maximum
        time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)
        perf_time.append(time_to_95)
    else:
        perf_time.append(None)

# plt.figure(figsize = (8,6))
# plt.scatter(nD_range, perf_val, c='black', marker = 'X')
# plt.plot(nD_range, perf_val, c='black')
# plt.xlabel("Number of distractors")
# plt.ylabel("Proportion under target shelter")
# plt.ylim([0, 1])
# plt.show()

# plt.figure(figsize = (8,6))
# plt.scatter(nD_range, perf_time, c='black', marker = 'X')
# plt.plot(nD_range, perf_time, c='black')
# plt.xlabel("Number of distractors")
# plt.ylabel("Time to 95% completion (s)")
# plt.ylim([0, 1100])
# plt.show()



model = "uncommitted_capacity"

nDmax=50
config="half_size_light"
if config == "half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)

target_proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{config}_1decay_target_proportions.npy")
time_constants = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{config}_1decay_time_constants.npy")
n_repeats = target_proportions.shape[1]

median_proportions = np.median(target_proportions,axis = 1)
sd_proportions = target_proportions.std(axis = 1)
mean_time_constants = time_constants.mean(axis = 1)
sd_time_constants = time_constants.std(axis = 1)


y_bins = 1000


x_values = np.repeat(nD_range, n_repeats)
y_values = np.linspace(0, 1, y_bins)  # Fixed y range for KDE
density_matrix = np.zeros((len(nD_range), y_bins))  # Store KDE results

for i, x in enumerate(nD_range):
    kde = gaussian_kde(target_proportions[i], bw_method=0.1)  # KDE for each x
    density_values = kde(y_values)
    density_matrix[i, :] = density_values / np.max(density_values)  

X_edges = np.arange(len(nD_range) + 1)  # 10 edges for 9 bins
Y_edges = np.linspace(0, 1, y_bins + 1)  # 1001 edges for 1000 bins
X, Y = np.meshgrid(X_edges, Y_edges)

# Set global font size
plt.rcParams.update({'font.size': 22})


fig=plt.figure(figsize=(14, 7))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')

plt.scatter(np.arange(len(nD_range)) + 0.5, median_proportions, c='black', marker='X', s=100)
plt.plot(np.arange(len(nD_range)) + 0.5, perf_val, c='tab:red', linewidth = 6)
plt.xticks(np.arange(len(nD_range)) + 0.5, nD_range)

plt.xlabel("Number of distractors")
plt.ylabel("Final proportion under target")
plt.ylim(0, 1)
plt.tight_layout()
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()

y_bins = 1000
y_values = np.linspace(0, 1100, y_bins)
density_matrix = np.zeros((len(nD_range), y_bins))

for i, x in enumerate(nD_range):
    valid_data = time_constants[i][~np.isnan(time_constants[i])]
    if len(valid_data) > 1:
        kde = gaussian_kde(valid_data, bw_method=0.1)
        density_values = kde(y_values)
        density_matrix[i, :] = density_values / np.max(density_values)
    else:
        density_matrix[i, :] = np.zeros_like(y_values)


X_edges = np.arange(len(nD_range) + 1)
Y_edges = np.linspace(0, 1100, y_bins + 1)
X, Y = np.meshgrid(X_edges, Y_edges)

fig=plt.figure(figsize=(14, 7))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')

mean_time_constants = np.nanmean(time_constants, axis=1)
plt.scatter(np.arange(len(nD_range)) + 0.5, mean_time_constants, c='black', marker='X', s=100)
plt.plot(np.arange(len(nD_range)) + 0.5, perf_time, linewidth = 6, c='tab:red')
plt.xticks(np.arange(len(nD_range)) + 0.5, nD_range)
plt.xlabel("Number of distractors")
plt.ylabel("$T_{95}$ (s)")
plt.ylim([-20, 1100])
plt.tight_layout()
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()