from multiprocessing import Pool, cpu_count
from cockroach_abm import *
from cockroach_abm_uncommitted import *
from cockroach_abm_uncommitted_capacity import *
from cockroach_abm_spatial import *
import numpy as np
from tqdm import tqdm
 
model = "uncommitted_capacity"





# PARAMETERS
N = 10
theta_base = 0.01
t_max = 1000000

n_repeats = 10
nD = 1

sizeHQ = 1.0
lightHQ = 1.0

# distractor_ranges = np.arange(1,13, 0.5)
distractor_ranges = np.arange(1, 2, 0.05)
    

parameters = {"h": 1, "rho": 1667}

# OUTPUT STORAGE
final_proportions = np.zeros((len(distractor_ranges), len(distractor_ranges), n_repeats))


def simulate_vectorized(N, nD, t_max, s, theta, parameters, dt=0.1, uncommitted_timescale=500): 
  rng = np.random.default_rng() # State encodings: 0..nD = shelters, nD+1 = uncommitted 
  states = np.full(N, nD+1, dtype=int) 
  # Precompute uncommitted transition probabilities 
  uncommitted_probability = np.exp(-dt / uncommitted_timescale) 
  uncommitted_probs = np.ones(nD+2) * (1 - uncommitted_probability) / (nD+1) 
  uncommitted_probs[-1] = uncommitted_probability 
  shelter_numbers_over_time = np.zeros((t_max, nD+2), dtype=int) 
  shelter_numbers_over_time[0] = np.bincount(states, minlength=nD+2) 
  for t in range(1, t_max): 
    counts = np.bincount(states, minlength=nD+2) 
    shelter_numbers_over_time[t] = counts 
    
    # --- Update committed agents (under shelters) --- 
    committed_mask = states <= nD 
    if committed_mask.any(): 
      dens = counts[states[committed_mask]] / s[states[committed_mask]] 
      q = parameters['h'] * theta[states[committed_mask]] / (1 + parameters['rho'] * dens**2) 
      p_stay = np.exp(-q * dt) 
      leave_mask = rng.random(p_stay.size) > p_stay 
      states[np.where(committed_mask)[0][leave_mask]] = nD+1 
      
    # --- Update uncommitted agents --- 
    uncommitted_mask = states == nD+1 
    if uncommitted_mask.any(): 
      choices = rng.choice(np.arange(nD+2), size=uncommitted_mask.sum(), p=uncommitted_probs) 
      
      # rejection rule: only apply to real shelters (0..nD) 
      reject_probs = np.zeros_like(choices, dtype=float) 
      shelter_mask = choices != nD+1 
      reject_probs[shelter_mask] = counts[choices[shelter_mask]] / s[choices[shelter_mask]] 
      reject = shelter_mask & (rng.random(choices.size) < reject_probs) 
      choices[reject] = nD+1 
      states[uncommitted_mask] = choices 
  return shelter_numbers_over_time

  
# SIMULATION FUNCTION (single repeat)
def run_single_repeat(args):
    nD, parameters, t_max, N, theta_base, sizeLQ, lightLQ = args

#    s = np.array([N * sizeLQ, N * sizeHQ])
#    theta = np.array([theta_base * lightHQ, theta_base * lightLQ])
    
    s = np.array([N, N * sizeLQ])
    theta = np.array([theta_base * 2, theta_base * lightLQ])
    
    

    # Run simulation
    shelter_numbers = simulate_vectorized(N, nD, t_max, s, theta, parameters)
  
    # Compute outputs
    
    shelter_picked = shelter_numbers[-10:].argmax()
    final_numbers = shelter_numbers[-10:].mean(axis=0)
#        if final_numbers[0] > final_numbers[1]:
#          proportion = final_numbers[0]
#        else:
#          proportion = -final_numbers[1]
    proportion = final_numbers[0]-final_numbers[1]
      
    return proportion



# Serial over sizeLQ, parallel over repeats
for i, sizeLQ in enumerate(tqdm(distractor_ranges, desc="sizeLQ values")):
    for j, lightLQ in enumerate(distractor_ranges):
      args_list = [(nD, parameters, t_max, N, theta_base, sizeLQ, lightLQ) for _ in range(n_repeats)]
  
      with Pool(cpu_count()) as pool:
          results = list(pool.imap_unordered(run_single_repeat, args_list))
  
      final_proportions[i, j, :] = results
np.save(f"Analysed_data/cockroach_abm_{model}/{N}_compensation_search_proportions.npy", final_proportions)

