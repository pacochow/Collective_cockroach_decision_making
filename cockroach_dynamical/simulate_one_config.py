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


distractor_type = "half_size_light"
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
N=100
nDmax=50
theta_base=0.5
rho=600
n=2
max_time=2000
dt = 0.01

if distractor_type=="half_size_light":
    nD_range = np.arange(2, nDmax, 2)
else:
    nD_range = np.arange(1, nDmax)


times = np.arange(0, max_time, dt)

perf_val = []
perf_time = []
for nD in nD_range:
    mu = 1 / (1 + nD)  # Probability of finding shelter

    if distractor_type=="half_size_light":
        s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    else:
        s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    

    # Pack parameters into a dictionary
    params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
    
    # Solve the ODE system
    def model(t, x): return ode_sys(t, x, params)
    x0 = np.ones(len(s))*N/len(s)  # Initial conditions
    sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
    max_val = sol.y[0, -1]
    max_idx = np.argmax(sol.y[0])

    perf_val.append(max_val/params['N'])
    if max_val/N > 0.5:
        # Find the time at which the solution reaches 95% of its maximum
        time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)
        perf_time.append(time_to_95)
    else:
        perf_time.append(None)


    # print(f"Number of distractors: {nD}", sol.y.shape)
    
    # for i in np.arange(1, nD+1)[::-1]:
    #     if i <= nD//2:
    #         plt.plot(times, sol.y[i], c='tab:green', label = "Low light quality distractor")
    #     else:
    #         plt.plot(times, sol.y[i], c='tab:blue', label = "Low size quality distractor")
    # plt.plot(times, sol.y[0], c='black', label = "Target")
    # plt.ylim([-5, N])
    # handles, labels = plt.gca().get_legend_handles_labels()
    # unique_labels = dict(zip(labels, handles))  # Keep only unique labels
    # plt.legend(unique_labels.values(), unique_labels.keys())
    # plt.xlabel("Time")
    # plt.ylabel("Number of individuals")
    # plt.show()

plt.figure(figsize = (8,6))
plt.scatter(nD_range, perf_val, c='black', marker = 'X')
plt.plot(nD_range, perf_val, c='black')
plt.xlabel("Number of distractors")
plt.ylabel("Proportion under target shelter")
plt.ylim([0, 1])
plt.show()

plt.figure(figsize = (8,6))
plt.scatter(nD_range, perf_time, c='black', marker = 'X')
plt.plot(nD_range, perf_time, c='black')
plt.xlabel("Number of distractors")
plt.ylabel("Time to 95% completion (s)")
plt.ylim([0, 1100])
plt.show()

# perf_val = []
# perf_time = []
# for i in range(nDmax // 2 + 1):
#     nD = i*2  # Current number of distractors

#     mu = 1 / (1 + nD)  # Probability of finding shelter
#     s = np.concatenate([[N], np.full(nD//2, config[0] * N), np.full(nD//2, config[1] * N)]) # Shelter capacities
#     theta = np.concatenate([[theta_base], np.full(nD//2, config[2] * theta_base), np.full(nD//2, config[3] * theta_base)]) # Shelter light levels
    
#     # Pack parameters into a dictionary
#     params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
    
#     # Solve the ODE system
#     def model(t, x): return ode_sys(t, x, params)
#     x0 = np.zeros(len(s))  # Initial conditions
#     sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)
#     if sol.y.shape[1] > 0:  # Check if the solution is valid
#         max_val = np.max(sol.y[0])
#         max_idx = np.argmax(sol.y[0])

#         perf_val.append(max_val/params['N'])

#         # Find the time at which the solution reaches 95% of its maximum
#         time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)
#         perf_time.append(time_to_95)
#     else:
#         perf_val.append(None)
#         perf_time.append(None)
    
#     print(f"Number of distractors: {nD}", sol.y.shape)
#     for i in range(nD+1):
#         plt.plot(times, sol.y[i])
#     plt.ylim([-5, 100])
#     plt.show()



# plt.plot(np.arange(0, nDmax + 1, 2), perf_val)
# plt.xlabel("Number of distractors")
# plt.ylabel("Proportion of individuals picking optimal shelter")
# plt.legend()
# plt.show()


# plt.plot(np.arange(0, nDmax + 1, 2), perf_time)
# plt.xlabel("Number of distractors")
# plt.ylabel("Time to 95% completion")
# plt.legend()
# plt.show()