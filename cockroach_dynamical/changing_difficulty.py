import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


distractor_type = "low_size"
sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0


N=100
nD=1
nDmax = 50
theta_base=0.01
rho=1667
n=2
max_time=2000
dt = 0.01

if distractor_type=="half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)

difficulty_range = np.arange(0.1, 2.6, 0.01)

times = np.arange(0, max_time, dt)

shelter1val = []
shelter2val = []
perf_time = []
for difficulty in difficulty_range:
    sizeLQ = difficulty
    configs = {
    "low_size": [sizeLQ, sizeLQ, lightHQ, lightHQ],
    "low_light": [sizeHQ, sizeHQ, lightLQ, lightLQ],
    "low_size_light": [sizeLQ, sizeLQ, lightLQ, lightLQ],
    "half_size_light": [sizeHQ, sizeLQ, lightLQ, lightHQ]}
    config = configs[distractor_type]
    # mu = 1 / (1 + nD)  # Probability of finding shelter
    mu = 0.001
    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        # s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        # theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
        s = np.array([config[0]*N, config[0]*N])
        theta = np.array([theta_base, config[2]*theta_base])

    # Pack parameters into a dictionary
    params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
    
    # Solve the ODE system
    def model(t, x): return ode_sys(t, x, params)
    # x0 = np.ones(len(s))*N/len(s)  # Initial conditions
    x0 = np.zeros(len(s))  # Initial conditions
    sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
    shelter_picked = sol.y[:, -1].argmax()
    max_val = sol.y[shelter_picked, -1]
    shelter1val.append(sol.y[0, -1]/params['N'])
    shelter2val.append(sol.y[1, -1]/params['N'])
    if max_val/N > 0.5:
        # Find the time at which the solution reaches 95% of its maximum
        time_to_95 = next((t for t, y in zip(times, sol.y[shelter_picked, :]) if y >= 0.95 * max_val), None)
        perf_time.append(time_to_95)
    else:
        perf_time.append(None)


plt.figure(figsize = (8,6))
plt.scatter(difficulty_range, shelter1val, c='tab:blue', marker = 'X')
plt.plot(difficulty_range, shelter1val, c='tab:blue', linewidth = 2)
plt.scatter(difficulty_range, shelter2val, c='tab:red', marker = 'X')
plt.plot(difficulty_range, shelter2val, c='tab:red', linewidth = 2)
plt.xlabel("Size of distractor")
plt.ylabel("Proportion under shelter")
plt.ylim([0, 1])
plt.show()

plt.figure(figsize = (8,6))
plt.scatter(difficulty_range, perf_time, c='black', marker = 'X')
plt.plot(difficulty_range, perf_time, c='black')
plt.xlabel("Number of distractors")
plt.ylabel("Time to 95% completion (s)")
plt.show()
