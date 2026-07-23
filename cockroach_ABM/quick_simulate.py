from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
from cockroach_abm_spatial import *
import matplotlib.pyplot as plt

distractor_type = "half_size_light"
theta_base = 0.01
t_max = 100000
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
nD = 4
parameters = {
    "h": 1,
    "rho": 1667
}

if distractor_type == "half_size_light":
    s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
else:
    s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels

trans = np.load(f"../Analysed_data/transition_matrices/{nD+1}_transition_probs.npy")

agents = [Cockroach_agent_uncommitted_capacity(nD, t_max) for _ in range(N)]
shelter_numbers = simulate(agents, s, theta, parameters, t_max = t_max)


for i in range(nD+1):
    if i == 0:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, i]/N, c= 'black', linewidth = 2)
    elif i > 0 and i <= (nD-1)/2+1:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, i]/N, c='tab:red', linewidth = 2)
    else:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, i]/N, c='tab:orange', linewidth = 2)
plt.xlabel("Time (s)")
plt.ylabel("Proportion of individuals \nunder shelter")
# plt.legend(["Target", "Distractor"])
plt.show()


# plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:, -1], c='black', linewidth = 4)
# plt.xlabel("Time (s)")
# plt.ylabel("Number of uncommitted cockroaches")
# # plt.ylim([0, 30])
# plt.show()

### Closeup
t_max = 2000
for i in range(nD+1):
    if i == 0:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:t_max, i]/N, c= 'black', linewidth = 2)
    elif i > 0 and i <= (nD+1)/2:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:t_max, i]/N, c='tab:red', linewidth = 2)
    else:
        plt.plot(np.arange(t_max)*agents[0].dt, shelter_numbers[:t_max, i]/N, c='tab:orange', linewidth = 2)
plt.xlabel("Time (s)")
plt.ylabel("Proportion of individuals \nunder shelter")
# plt.legend(["Target", "Distractor"])
plt.show()
