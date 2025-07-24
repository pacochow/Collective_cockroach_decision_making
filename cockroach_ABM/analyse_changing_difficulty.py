from cockroach_abm_uncommitted_capacity import *
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm
import matplotlib.pyplot as plt


model = "uncommitted_capacity"
theta_base = 0.5
t_max = 10000

N = 100
nD = 1
parameters = {
    "h": 1,
    "rho": 600
}

difficulty_range = np.arange(0.4, 2.6, 0.1)
n_difficulty_range = len(difficulty_range)

n_repeats = 500
target_proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/changing_difficulty_target_proportions.npy")
time_constants = np.load(f"../Analysed_data/cockroach_abm_{model}/changing_difficulty_time_constants.npy")



median_proportions = np.median(target_proportions,axis = 1)
sd_proportions = target_proportions.std(axis = 1)
mean_time_constants = time_constants.mean(axis = 1)
sd_time_constants = time_constants.std(axis = 1)


y_bins = 1000

x_values = np.repeat(difficulty_range, n_repeats)
y_values = np.linspace(0, 1, y_bins)  # Fixed y range for KDE
density_matrix = np.zeros((len(difficulty_range), y_bins))  # Store KDE results

for i, x in enumerate(difficulty_range):
    kde = gaussian_kde(target_proportions[i], bw_method=0.1)  # KDE for each x
    density_values = kde(y_values)
    density_matrix[i, :] = density_values / np.max(density_values)  

X_edges = np.arange(len(difficulty_range) + 1)  # 10 edges for 9 bins
Y_edges = np.linspace(0, 1, y_bins + 1)  # 1001 edges for 1000 bins
X, Y = np.meshgrid(X_edges, Y_edges)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')
plt.scatter(np.arange(len(difficulty_range)) + 0.5, median_proportions, c='black', marker='X', s=100)
ticks = np.array([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5])
labels = np.round(((ticks-0.5)/10+0.4), 1)
plt.xticks(ticks, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Difficulty", fontsize=16)
plt.ylabel("Proportion under target shelter", fontsize=16)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

y_bins = 1000
y_values = np.linspace(0, 1100, y_bins)
density_matrix = np.zeros((len(difficulty_range), y_bins))

for i, x in enumerate(difficulty_range):
    valid_data = time_constants[i][~np.isnan(time_constants[i])]
    if len(valid_data) > 1:
        kde = gaussian_kde(valid_data, bw_method=0.1)
        density_values = kde(y_values)
        density_matrix[i, :] = density_values / np.max(density_values)
    else:
        density_matrix[i, :] = np.zeros_like(y_values)


X_edges = np.arange(len(difficulty_range) + 1)
Y_edges = np.linspace(0, 1100, y_bins + 1)
X, Y = np.meshgrid(X_edges, Y_edges)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, density_matrix.T, cmap="Blues", shading='auto')

mean_time_constants = np.nanmean(time_constants, axis=1)
plt.scatter(np.arange(len(difficulty_range)) + 0.5, mean_time_constants, c='black', marker='X', s=100)

plt.xticks(ticks, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Number of distractors", fontsize=16)
plt.ylabel("Time to 95% completion (s)", fontsize=16)
plt.ylim([-20, 1100])
plt.tight_layout()
plt.show()