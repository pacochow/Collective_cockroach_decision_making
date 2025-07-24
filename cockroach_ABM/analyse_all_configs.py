import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter


model = "uncommitted_capacity"

configs = ["low_size", "low_light", "low_size_light", "half_size_light"]
labels = ["Low size quality", "Low light quality", "Low size and light quality", "1/2 low size and 1/2 low light quality"]
colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
nDmax = 50
n_iterations = 500
y_bins = 1000



# PLOT ALL CONFIGS
plt.figure(figsize=(6, 6))
for i, config in enumerate(configs):
    target_proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{config}_1decay_target_proportions.npy")
    mean_proportions = np.nanmedian(target_proportions,axis = 1)
    if config == "half_size_light":
        nD_range = np.arange(2, nDmax, 2)
    else:
        nD_range = np.arange(1, nDmax)
  
    plt.scatter(nD_range, mean_proportions, c = colors[i], marker='X', alpha = 0.2)
    plt.plot(nD_range, savgol_filter(mean_proportions, window_length = 5, polyorder = 2), c=colors[i], linewidth = 4, label = labels[i])
plt.xticks([0, 10, 20, 30, 40, 50], size = 18)
plt.yticks(size=18)
plt.xlabel("Number of distractors", size = 18)
plt.ylabel("Proportion under target shelter", size = 18)
plt.ylim([-0.1, 1.1])
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
for i, config in enumerate(configs):
    time_constants = np.load(f"../Analysed_data/cockroach_abm_{model}/100_{config}_1decay_time_constants.npy")
    if config == "half_size_light":
        nD_range = np.arange(2, nDmax, 2)
    else:
        nD_range = np.arange(1, nDmax)

    mean_time_constants = np.nanmean(time_constants,axis = 1)

    plt.scatter(nD_range, mean_time_constants, c = colors[i], marker='X', alpha = 0.2)
    plt.plot(nD_range, savgol_filter(mean_time_constants, window_length = 5, polyorder = 2), c=colors[i], linewidth = 4, label = labels[i])
plt.xlabel("Number of distractors", size = 18)
plt.xticks([0, 10, 20, 30, 40, 50], size = 18)
plt.yticks([0, 200, 400, 600, 800, 1000], size = 18)
plt.ylabel("Time to 95% completion (s)", size = 18)
plt.legend()
plt.show()

# CHECK INDIVIDUAL CONFIGS
# for config in configs:
#     target_proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/{config}_1decay_target_proportions.npy")
#     mean_proportions = target_proportions.mean(axis = 1)
#     if config == "half_size_light":
#         nD_range = np.arange(2, nDmax, 2)
#     else:
#         nD_range = np.arange(1, nDmax)
#     x_values = np.repeat(nD_range, target_proportions.shape[1])

#     y_values = np.linspace(0, 1, y_bins)  # Fixed y range for KDE
#     density_matrix = np.zeros((len(nD_range), y_bins))  # Store KDE results

#     for i, x in enumerate(nD_range):
#         kde = gaussian_kde(target_proportions[i], bw_method=0.1)  # KDE for each x
#         density_values = kde(y_values)
#         density_matrix[i, :] = density_values / np.max(density_values)  

#     plt.figure(figsize=(8, 6))
#     plt.imshow(density_matrix.T, aspect="auto", origin="lower",
#                extent=[nD_range.min(), nD_range.max(), y_values.min(), y_values.max()],
#                cmap="Blues", alpha = 0.7)
#     plt.scatter(nD_range, mean_proportions, c='black', marker='X')
#     plt.xticks([0, 20, 40, 60, 80, 100])
#     plt.xlabel("Number of distractors")
#     plt.ylabel("Proportion under target shelter")
#     plt.ylim([-0.1, 1.1])
#     plt.show()



#     time_constants = np.load(f"../Analysed_data/cockroach_abm_{model}/{config}_1decay_time_constants.npy")
#     y_values = time_constants.flatten()
#     mean_time_constants = time_constants.mean(axis = 1)
#     sd_time_constants = time_constants.std(axis = 1)
#     y_values = np.linspace(0, 1100, y_bins)  # Fixed y range for KDE   
#     density_matrix = np.zeros((len(nD_range), y_bins))  # Store KDE results

#     for i, x in enumerate(nD_range):
#         kde = gaussian_kde(time_constants[i], bw_method=0.1)  # KDE for each x
#         density_values = kde(y_values)

#         density_matrix[i, :] = density_values / np.max(density_values)  

#     plt.figure(figsize=(8, 6))
#     plt.imshow(density_matrix.T, aspect="auto", origin="lower",
#                extent=[nD_range.min(), nD_range.max(), y_values.min(), y_values.max()],
#                cmap="Blues")
#     plt.scatter(nD_range, mean_time_constants, c='black', marker='X')
#     plt.xlabel("Number of distractors")
#     plt.xticks([0, 20, 40, 60, 80, 100])
#     plt.ylim([-20, 1100])
#     plt.ylabel("Time to 95% completion (s)")
#     plt.show()
