from multiprocessing import Pool, cpu_count
from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
import numpy as np
from tqdm import tqdm

model = "uncommitted_capacity"
distractor_type = "low_size"

# PARAMETERS
N = 100
theta_base = 0.5
t_max = 10000
n_repeats = 500
nDmax = 50
if distractor_type=="half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)
parameters = {"h": 1, "rho": 600}
config = [1.0, 1.75, 1.75, 1.0] 

# OUTPUT STORAGE
target_proportions = np.zeros((len(nD_range), n_repeats))
time_constants = np.zeros((len(nD_range), n_repeats))

# SIMULATION FUNCTION (single repeat)
def run_single_repeat(args):
    nD, config, parameters, t_max, N, theta_base = args

    # Set up shelters
    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    

    # Run simulation
    agents = [Cockroach_agent_uncommitted_capacity(nD, t_max) for _ in range(N)]
    shelter_numbers = simulate(agents, s, theta, parameters, t_max=t_max)

    # Compute outputs
    shelter_picked = shelter_numbers[-1].argmax()
    picked_number = shelter_numbers[-10:, shelter_picked].mean()

    target_prop = shelter_numbers[-10:, 0].mean() / N
    if target_prop < 0.5:
        time_to_95 = np.nan
    else:
        time_to_95 = (shelter_numbers[:, shelter_picked] >= 0.95 * picked_number).argmax() * agents[0].dt

    return target_prop, time_to_95


# MAIN LOOP: Serial over nD, parallel over repeats
for i, nD in enumerate(tqdm(nD_range, desc="nD values")):
    args_list = [(nD, config, parameters, t_max, N, theta_base) for _ in range(n_repeats)]

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(run_single_repeat, args_list))

    target_proportions[i, :] = [r[0] for r in results]
    time_constants[i, :] = [r[1] for r in results]
np.save(f"Analysed_data/cockroach_abm_{model}/{distractor_type}_1decay_target_proportions.npy", target_proportions)
np.save(f"Analysed_data/cockroach_abm_{model}/{distractor_type}_1decay_time_constants.npy", time_constants)

