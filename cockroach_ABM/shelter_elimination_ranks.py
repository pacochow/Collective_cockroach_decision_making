from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
import matplotlib.pyplot as plt 
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *


model = "uncommitted_capacity"
distractor_type = "half_size_light"
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

N = 100
nD = 6
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

# n_iterations = 10
# eliminated_ranks = {key: [] for key in range(nD+1)}


# for _ in tqdm(range(n_iterations)):
#     agents = [Cockroach_agent_uncommitted_capacity(nD, t_max) for _ in range(N)]
#     shelter_numbers = simulate(agents, s, theta, parameters, t_max = t_max)

#     max_times = []
#     for i in range(nD+1):
#         shelter = shelter_numbers[:, i]
#         max_times.append(np.argmax(shelter))
#     elimination_rank = np.argsort(max_times)
#     for j in range(nD+1):
#         eliminated_ranks[j].append(elimination_rank[j])
    
# for i in eliminated_ranks:
#     eliminated_ranks[i] = [0 if j==0 else 1 if j in np.arange(1, nD/2+1) else 2 for j in eliminated_ranks[i]]

# x_vals = np.arange(nD+1)  
# cat_unique = [0, 1, 2]


# counts = np.zeros((len(cat_unique), len(x_vals)))
# for j, x in enumerate(x_vals):
#     for i, cat in enumerate(cat_unique):
#         counts[i, j] = eliminated_ranks[x].count(cat)


# proportions = counts / counts.sum(axis=0, keepdims=True)

# np.save(f"Analysed_data/cockroach_abm_{model}/elimination_proportions.npy", proportions)


x_vals = np.arange(nD+1)  
cat_unique = [0, 1, 2]
proportions = np.load(f"../Analysed_data/cockroach_abm_{model}/elimination_proportions_{nD}.npy")

# Set global font size
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(8, 6))
bottom = np.zeros(len(x_vals))
colors = ['black', 'tab:red', 'tab:orange']
labels = ['Target', 'Bright distractor', 'Large distractor']

for i in range(len(cat_unique)):
    ax.bar(x_vals, proportions[i], bottom=bottom, label=labels[i], color=colors[i])
    bottom += proportions[i]

ax.set_xlabel("Elimination order")
ax.set_ylabel("Proportion")
ax.set_xticks(x_vals, x_vals+1)
# ax.legend()
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()