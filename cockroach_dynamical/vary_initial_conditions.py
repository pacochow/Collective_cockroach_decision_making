import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to define the ODE system
def ode_sys(t, x, p):
    """
    Compute the rate of change for the ODE system.
    
    Parameters:
        t (float): Time.
        x (array): Current state of the system (individuals in each shelter).
        p (dict): Dictionary containing parameters (s, theta, mu, rho, n, N).
        
    Returns:
        dx (array): Rate of change of the system state.
    """
    D = x / p['s']  # Compute densities
    Q = p['theta'] / (1 + p['rho'] * D**p['n'])  # Light-influence function
    dx = -x * Q + p['mu'] * (1 - D) * (p['N'] - np.sum(x))  # Rate of change
    return dx

sizeLQ = 1.5
sizeHQ = 1.0
lightLQ = 1.5
lightHQ = 1.0
config = [sizeLQ, sizeHQ, lightHQ, lightLQ] 
N=100
nDmax=50
theta_base=0.5
rho=600
n=2
max_time=2000
dt = 0.01


times = np.arange(0, max_time, dt)


perf_val = []
perf_time = []




nD = 24  # Current number of distractors
mu = 1 / (1 + nD)  # Probability of finding shelter
results = np.zeros((nD, 20))
for nOTargets in tqdm(range(nD)):
    s = np.concatenate([[N], np.full(nOTargets, config[0] * N), np.full(nD-nOTargets, config[1] * N)]) # Shelter capacities
    theta = np.concatenate([[theta_base], np.full(nOTargets, config[2] * theta_base), np.full(nD-nOTargets, config[3] * theta_base)]) # Shelter light levels
        

    for i in range(20):
        # Pack parameters into a dictionary
        params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}


        # Solve the ODE system
        def model(t, x): return ode_sys(t, x, params)
        x0 = np.zeros(len(s))  # Initial conditions
        x0[0] = i
        sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
        max_val = sol.y[0, -1]

        results[nOTargets, i] = max_val
        # plt.plot(times, sol.y[0], c='black', label = "Target")
        # for i in np.arange(1, nD+1)[::-1]:
        #     if i <= nD//2:
        #         plt.plot(times, sol.y[i], c='tab:green', label = "Other targets")
        #     else:
        #         plt.plot(times, sol.y[i], c='tab:blue', label = "Low size quality distractor")
        # print(i, s, theta)
        # plt.ylim([-5, N])
        # handles, labels = plt.gca().get_legend_handles_labels()
        # unique_labels = dict(zip(labels, handles))  # Keep only unique labels
        # plt.legend(unique_labels.values(), unique_labels.keys())
        # plt.xlabel("Time")
        # plt.ylabel("Number of individuals")
        # plt.show()


plt.imshow(results, cmap = 'summer')
plt.ylabel("Number of targets")
plt.yticks([0, 4, 9, 14, 19],labels=[1, 5, 10, 15, 20])
plt.xlabel("Initial number of individuals at a target")
plt.colorbar(label = 'Final number of individuals at seeeded target')
plt.show()