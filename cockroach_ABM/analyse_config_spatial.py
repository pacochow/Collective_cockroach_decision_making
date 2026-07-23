import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.integrate import solve_ivp


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
nDmax=28
theta_base=0.01
rho=1667
n=2
max_time=2000000
dt = 0.1

nD_range = np.arange(2, nDmax, 2)


times = np.arange(0, max_time, dt)

perf_val = []
perf_time = []
for nD in nD_range:
    mu = 0.002/(1+nD)
    theta_T = theta_base
    theta_L = theta_base*lightHQ
    theta_B = theta_base*lightLQ
    theta_D = theta_base*config[2]

    S_T = N
    S_L = N*sizeLQ
    S_B = N*sizeHQ
    S_D = N*config[0]

    ns =nD+1
    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    

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

    x0 = np.zeros(len(s))  # Initial conditions
    if distractor_type != "half_size_light":
        x0 = np.array([0.0, 0.0])   # start uncommitted
        sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
    else:
        x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
        sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

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



model = "spatial"


nD_range = np.arange(2, nDmax, 2)

target_proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{distractor_type}_500decay_target_proportions.npy")[:13]
time_constants = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{distractor_type}_500decay_time_constants.npy")[:13]
n_repeats = target_proportions.shape[1]

median_proportions = np.median(target_proportions,axis = 1)
sd_proportions = target_proportions.std(axis = 1)
counts = np.sum(~np.isnan(time_constants), axis=1)
median_time_constants = np.nanmedian(time_constants,axis = 1)
median_time_constants[counts<=10] = np.nan
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
plt.rcParams.update({'font.size': 24})


fig=plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]

pcm = ax.pcolormesh(X, Y, density_matrix.T, cmap="Greys", shading='auto')
ax.plot(np.arange(len(nD_range)) + 0.5, perf_val, c='dodgerblue', linewidth=6, linestyle = '--')
ax.scatter(np.arange(len(nD_range)) + 0.5, median_proportions, c='dodgerblue', marker='X', s=150, edgecolor ='black')

xticks = np.arange(len(nD_range)) + 0.5
ax.set_xticks(xticks)


desired_labels = [2, 8, 14, 20, 26]
xtick_labels = [str(d) if d in desired_labels else "" for d in nD_range]
ax.set_xticklabels(xtick_labels)


ax.set_xlabel("Number of distractors")
ax.set_ylabel("Final proportion \nunder target")
ax.set_ylim(0, 1)

for spine in ax.spines.values():
    spine.set_linewidth(2)
#plt.savefig("../figs/spatial_abm_conjunction_search_accuracy.png", bbox_inches = 'tight')
plt.show()

y_bins = 1000
y_values = np.linspace(0, 700100, y_bins)
density_matrix = np.zeros((len(nD_range), y_bins))

for i, x in enumerate(nD_range):
    if counts[i]>10:
        valid_data = time_constants[i][~np.isnan(time_constants[i])]
        if len(valid_data) > 1:
            kde = gaussian_kde(valid_data, bw_method=0.1)
            density_values = kde(y_values)
            density_matrix[i, :] = density_values / np.max(density_values)
        else:
            density_matrix[i, :] = np.zeros_like(y_values)


X_edges = np.arange(len(nD_range) + 1)
Y_edges = np.linspace(0, 700100, y_bins + 1)
X, Y = np.meshgrid(X_edges, Y_edges)

fig=plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # Same position as above

pcm = ax.pcolormesh(X, Y, density_matrix.T, cmap="Greys", shading='auto')
ax.plot(np.arange(len(nD_range)) + 0.5, perf_time, linewidth=6, c='dodgerblue', linestyle = '--')
ax.scatter(np.arange(len(nD_range)) + 0.5, median_time_constants, c='dodgerblue', marker='X', s=150, edgecolor ='black')

xticks = np.arange(len(nD_range)) + 0.5
ax.set_xticks(xticks)



desired_labels = [2, 8, 14, 20, 26]
xtick_labels = [str(d) if d in desired_labels else "" for d in nD_range]
ax.set_xticklabels(xtick_labels)


ax.set_xlabel("Number of distractors")
ax.set_ylabel("$T_{95}$ (hrs)")
ax.set_ylim([-20, 700000])
hours = np.array([0, 24, 48, 72, 96, 120, 144, 168])
seconds = hours*3600
ax.set_yticks(seconds)
ax.set_yticklabels(hours)

for spine in ax.spines.values():
    spine.set_linewidth(2)
#plt.savefig("../figs/spatial_abm_conjunction_search_time.png", bbox_inches = 'tight')
plt.show()