from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde

model = "uncommitted_capacity"
distractor_type = "low_size"

N = 100
theta_base = 0.5
t_max = 10000
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

nDmax = 50
n_repeats = 500
parameters = {
    "h": 1,
    "rho": 600
}

if distractor_type=="half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)

num_nD = len(nD_range)
target_proportions = np.zeros((num_nD, n_repeats))
time_constants = np.zeros((num_nD, n_repeats))





for idx, nD in tqdm(enumerate(nD_range), total = num_nD):

    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    



    for k in range(n_repeats):
        agents = [Cockroach_agent_basic(nD, t_max)
            for _ in range(N)]
        shelter_numbers = simulate(agents, s, theta, parameters, t_max = t_max)
        
        shelter_picked = shelter_numbers[-1].argmax()
        picked_number = shelter_numbers[-10:, shelter_picked].mean()

        target_proportions[idx, k] = shelter_numbers[-10:, 0].mean()/N
        if shelter_numbers[-10:, 0].mean()/N < 0.5:
            time_to_95 = np.nan
        else:
            time_to_95 = (shelter_numbers[:, shelter_picked]>=0.95*picked_number).argmax()*agents[0].dt
        time_constants[idx, k] = time_to_95
        group_choice = np.argmax(shelter_numbers[-1])



# np.save(f"../Analysed_data/cockroach_abm_{model}/{distractor_type}_decay_target_proportions.npy", target_proportions)
# np.save(f"../Analysed_data/cockroach_abm_{model}/{distractor_type}_decay_time_constants.npy", time_constants)


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

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')
plt.scatter(np.arange(len(nD_range)) + 0.5, median_proportions, c='black', marker='X')

plt.xticks(np.arange(len(nD_range)) + 0.5, nD_range)  # Label centers
plt.xlabel("Number of distractors")
plt.ylabel("Proportion under target shelter")
plt.ylim(0, 1)
plt.show()



# Recompute y range
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
        density_matrix[i, :] = np.zeros_like(y_values)  # fallback if no valid data

# Create grid
X_edges = np.arange(len(nD_range) + 1)
Y_edges = np.linspace(0, 1100, y_bins + 1)
X, Y = np.meshgrid(X_edges, Y_edges)

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')

mean_time_constants = np.nanmean(time_constants, axis=1)
plt.scatter(np.arange(len(nD_range)) + 0.5, mean_time_constants, c='black', marker='X')

plt.xticks(np.arange(len(nD_range)) + 0.5, nD_range)
plt.xlabel("Number of distractors")
plt.ylabel("Time to 95% completion (s)")
plt.ylim([-20, 1100])
plt.show()