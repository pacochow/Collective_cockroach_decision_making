from cockroach_abm import *
from cockroach_abm_uncommitted_capacity import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


model = "uncommitted_capacity"
theta_base = 0.5
t_max = 10000
sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0
config = [sizeHQ, sizeLQ, lightLQ, lightHQ] 


parameters = {
    "h": 1,
    "rho": 600
}

n_iterations = 1000
nD = 42
shelter_over_time = np.zeros((n_iterations, t_max))
shelter_picked = np.zeros(n_iterations)




N = 100


s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels



# for k in tqdm(range(n_iterations)):
#     agents = [Cockroach_agent_uncommitted_capacity(nD, t_max)
#         for _ in range(N)]
#     shelter_numbers = simulate(agents, s, theta, parameters, t_max = t_max)
#     shelter_over_time[k, :] = shelter_numbers[:, 0]
#     shelter_picked[k] = shelter_numbers[-1].argmax()


# heatmap = np.zeros((N+1, t_max))
# for t in range(t_max):
#     for n in range(N+1):
#         n_success = np.sum(np.array((shelter_over_time[:, t]==n), dtype = bool) & np.array(shelter_picked==0, dtype = bool))
#         if n_success>0:
#             heatmap[n, t] = n_success/(shelter_over_time[:, t]==n).sum()

heatmap = np.load(f"../Analysed_data/cockroach_abm_{model}/heatmap.npy")


# Set global font size
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize = (6,6))
plt.imshow(heatmap, aspect = 'auto', origin = 'lower')
plt.xlabel("Time (s)")
plt.ylabel("Number of individuals \nunder target shelter")
cbar = plt.colorbar()
cbar.set_label("Probability of picking target \nshelter")
plt.yticks(np.arange(0, 110, 50))
plt.xticks(np.arange(0, 12000, 2000), np.arange(0, 1200, 200))
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()

plt.imshow(heatmap==1, aspect = 'auto', origin = 'lower')
plt.xlabel("Time (s)")
plt.ylabel("Number of individuals \nunder target shelter")
cbar = plt.colorbar()
cbar.set_label("Definitely picking target shelter")
plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(0, 12000, 2000), np.arange(0, 1200, 200))
plt.show()

# np.save(f"../Analysed_data/cockroach_abm_{model}/heatmap.npy", heatmap)

# distractor_type = "half_size_light"
# np.save(f"../Analysed_data/{distractor_type}_target_proportions.npy", target_proportions)


