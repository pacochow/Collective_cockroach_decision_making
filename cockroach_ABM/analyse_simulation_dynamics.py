import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *


model = "uncommitted_capacity"
t_max = 10000
nD = 6
target_median = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_target_median.npy")
bright_distractor_median = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_bright_distractor_median.npy")
large_distractor_median = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_large_distractor_median.npy")

# target_lower = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_target_lower.npy")
# bright_distractor_lower = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_bright_distractor_lower.npy")
# large_distractor_lower = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_large_distractor_lower.npy")

# target_upper = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_target_upper.npy")
# bright_distractor_upper = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_bright_distractor_upper.npy")
# large_distractor_upper = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_large_distractor_upper.npy")

shelter_numbers = np.load(f"../Analysed_data/cockroach_abm_{model}/half_size_light_6_shelter_numbers.npy")


# Set global font size
plt.rcParams.update({'font.size': 22})
fig=plt.figure(figsize = (6,6))

for i in range(200):
    plt.plot(np.arange(t_max),shelter_numbers[i, :, 0], alpha = 0.01, c='black')
    plt.plot(np.arange(t_max),shelter_numbers[i, :, 1:4], alpha = 0.01, c='tab:red')
    plt.plot(np.arange(t_max),shelter_numbers[i, :, 4:7], alpha = 0.01, c='tab:orange')
plt.plot(np.arange(t_max), target_median, c='black', linewidth = 4)
plt.plot(np.arange(t_max), bright_distractor_median, c='tab:red', linewidth = 4)
plt.plot(np.arange(t_max), large_distractor_median, c='tab:orange', linewidth = 4)
# plt.fill_between(np.arange(t_max), target_upper, target_lower, color='tab:blue', alpha = 0.1)
# plt.fill_between(np.arange(t_max), bright_distractor_upper, bright_distractor_lower, color='tab:red', alpha = 0.1)
# plt.fill_between(np.arange(t_max), large_distractor_upper, large_distractor_lower, color='tab:orange', alpha = 0.1)
plt.yticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xticks(np.arange(0, 12000, 2000), np.arange(0, 1200, 200))
plt.ylabel("Proportion uner each shelter")
plt.xlabel("Time (s)")
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()