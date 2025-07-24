from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *


model = "uncommitted_capacity"
distractor_type = "half_size_light"
theta_base = 0.5
t_max = 20000
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

N = 100
nD = 10
parameters = {
    "h": 1,
    "rho": 600
}

if distractor_type == "half_size_light":
    s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
else:
    s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels


agents = [Cockroach_agent_uncommitted_capacity(nD, t_max) for _ in range(N)]
shelter_numbers = simulate(agents, s, theta, parameters, t_max = t_max)


zeros = []
competing = []
for i in range(t_max):
    zeros.append((shelter_numbers[i]==0).sum())
    competing.append((shelter_numbers[i]>=20).sum())

poisson_means = np.zeros(shelter_numbers.shape[0])
for t in range(shelter_numbers.shape[0]):
    data = shelter_numbers[t, 1:]
    poisson_means[t] = np.mean(data)

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.subplots_adjust(hspace=0.3)

time = np.arange(t_max) * agents[0].dt

# Plot 1: Proportion under shelter
for i in range(nD+1):
    if i == 0:
        axs[0].plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, i]/N, c= 'tab:blue', linewidth = 3)
    else:
        axs[0].plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, i]/N, c='tab:orange', linewidth = 3)
axs[0].set_ylabel("Proportion of individuals under shelter")
axs[0].legend(["Target", "Distractor"])

# Plot 2: Number of competing distractor shelters
axs[1].scatter(time, competing, s=2, color='black')
axs[1].set_ylabel("Number of shelters \nwith over 20 individuals")
axs[1].set_yticks([0, 1, 2, 3])

axs[2].plot(time, poisson_means, linewidth=3, color='black')
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Mean number of individuals \nunder distractor shelters")

plt.tight_layout()
plt.show()




# # Fit exponential decay
# plateau_value = np.max(poisson_means[:2000])  # or use np.mean of initial chunk
# threshold = 0.90 * plateau_value

# # Find first time when it drops below threshold
# decay_start_index = np.argmax(poisson_means < threshold)
# decay_start_time = time[decay_start_index]



# # Define exponential decay model
# def exp_decay(t, lambda0, tau):
#     return lambda0 * np.exp(-t / tau)

# # Data after the start of decay
# decay_time = time[decay_start_index:]
# decay_values = poisson_means[decay_start_index:]

# # Fit the exponential
# popt, _ = curve_fit(exp_decay, decay_time, decay_values, p0=(plateau_value, 100))
# lambda0_fit, tau_fit = popt

# plt.figure(figsize=(10, 5))
# plt.plot(time, poisson_means, label='Data', linewidth=2)
# plt.plot(decay_time, exp_decay(decay_time, *popt), 'r--', label=f'Exponential fit (from {decay_start_time:.1f}s)\nτ = {tau_fit:.2f}s', linewidth=2)
# plt.axvline(decay_start_time, color='gray', linestyle=':', label='Decay Start')
# plt.xlabel("Time (s)")
# plt.ylabel("Mean number of individuals under distractor shelters")
# plt.legend()
# plt.tight_layout()
# plt.show()

# for i in range(1000):
#     if i %100 == 0:
#         plt.bar(np.arange(len(shelter_numbers[0])), np.sort(shelter_numbers[i])[::-1], color='black')
#         plt.xlabel("Shelters ranked by popularity")
#         plt.ylabel("Number of individuals under shelter")
#         plt.show()

# animate_shelter_numbers(shelter_numbers[:5000], f"shelter_dynamics_{model}.mp4", animation_time = 10)