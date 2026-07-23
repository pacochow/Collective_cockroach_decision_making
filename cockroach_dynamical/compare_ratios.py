import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0
config1 = [sizeLQ, sizeLQ, lightHQ, lightHQ] 
config2 = [sizeHQ, sizeHQ, lightLQ, lightLQ]
config3 = [sizeLQ, sizeHQ, lightHQ, lightLQ]
N=100
nD = 8
theta_base=0.01
rho=1667
n=2
max_time=100000
dt = 0.1





all_perf_vals = []
all_perf_times = []
labels = ["Low size quality", "3/4 low size + 1/4 low light quality", "1/2 low size + 1/2 low light quality", "1/4 low size + 3/4 low light quality", "Low light quality"]
colors = ['tab:orange', 'lightsteelblue', 'dodgerblue', 'navy', 'tab:red']
perf_val = []
perf_time = []
for i, proportion in enumerate(np.arange(5)):

    theta_T = theta_base
    theta_L = theta_base*lightHQ
    theta_B = theta_base*lightLQ

    S_T = N
    S_L = N*sizeLQ
    S_B = N*sizeHQ
    

    if i ==0:
        theta_D = theta_base*lightHQ
        S_D = N*sizeLQ
    elif i==4:
        theta_D = theta_base*lightLQ
        S_D = N*sizeHQ


    ns = nD+1
    mu = 0.002 / ns  # Probability of finding shelter
    


    nLightDistractor = nD*proportion//4



    def one_distractor_model(t, xyz):
        X, YD = xyz
        U = N-X-YD
        QT = theta_T / (1.0 + rho * ( (X / S_T)**n ))
        QD = theta_D / (1.0 + rho * ( ((YD / nD) / S_D)**n ))
        dx = -X * QT + mu * (1.0 - X / S_T) * U
        dy = -YD * QD + (nD * mu) * (1.0 - YD / (nD * S_D)) * U
        return np.array([dx, dy])


    def two_distractor_model(t, xyz):
        X, YL, YB = xyz
        U = N - X - YL - YB

        # Leaving terms (with density effects)
        QT = theta_T / (1.0 + rho * (X / S_T) ** n)
        QL = theta_L / (1.0 + rho * (4.0 * YL / ((4-proportion) * (ns - 1) * S_L)) ** n)
        QB = theta_B / (1.0 + rho * (4.0 * YB / (proportion * (ns - 1) * S_B)) ** n)

        alpha_T  = mu
        alpha_DL = mu * (ns - 1) * (4-proportion) / 4
        alpha_DB = mu * (ns - 1) * proportion/ 4

        dX  = -X  * QT + alpha_T  * (1.0 - X  / S_T) * U
        dYL = -YL * QL + alpha_DL * (1.0 - 4.0 * YL / ((4-proportion) * (ns - 1) * S_L)) * U
        dYB = -YB * QB + alpha_DB * (1.0 - 4.0 * YB / (proportion * (ns - 1) * S_B)) * U
        return np.array([dX, dYL, dYB])

    times = np.arange(0.0, max_time, dt)
    
    if i==0 or i == 4:
        x0 = np.array([0.0, 0.0])   # start uncommitted
        sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
    else:
        x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
        sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

    max_val = np.max(sol.y[0])
    max_idx = np.argmax(sol.y[0])

    perf_val.append(max_val/N)
    if max_val/N > 0.5:
        # Find the time at which the solution reaches 95% of its maximum
        time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)

        perf_time.append(time_to_95)
    else:
        perf_time.append(np.nan)

plt.rcParams.update({'font.size': 24})
# fig = plt.figure(figsize = (10,3))
fig = plt.figure(figsize = (21,2))
plt.bar(np.arange(5), perf_time, color='black')
# ticks = np.array([29000, 30000, 31000])
# plt.yticks(ticks, np.round(ticks/3600, 2))
hours = np.array([8, 8.7])
# hours = np.array([0, 14])
seconds = hours*3600
plt.yticks(seconds, hours)
plt.ylabel("$T_{95}$ (hrs)")
# plt.ylabel("Decision time (hrs)")
plt.ylim(seconds)
plt.xticks([0, 1, 2, 3, 4], ["4:0", "3:1", "2:2", "1:3", "0:4"])
# plt.xticks([0, 1, 2, 3, 4], [])
plt.xlabel("Ratio of large:bright distractors")

ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.savefig(f"../figs/{nD}_distractor_ratios.png", bbox_inches = 'tight')

plt.show()







# nDmax = 12
# all_perf_times = np.zeros((5, nDmax))
# fig = plt.figure(figsize = (14,6))
# all_perf_vals = []
# labels = ["Low size quality", "3/4 low size + 1/4 low light quality", "1/2 low size + 1/2 low light quality", "1/4 low size + 3/4 low light quality", "Low light quality"]
# colors = ['tab:orange', 'lightsteelblue', 'dodgerblue', 'navy', 'tab:red']

# for i, proportion in enumerate(np.arange(5)):
#     perf_val = []
#     perf_time = []
#     if proportion == 0 or proportion == 4:
#         nD_range = np.arange(1, nDmax+1)
#     elif proportion == 2:
#         nD_range = np.arange(2, nDmax+2, 2)
#     else:
#         nD_range = np.arange(4, nDmax+4, 4)
#     theta_T = theta_base
#     theta_L = theta_base*lightHQ
#     theta_B = theta_base*lightLQ

#     S_T = N
#     S_L = N*sizeLQ
#     S_B = N*sizeHQ
    

#     if i ==0:
#         theta_D = theta_base*lightHQ
#         S_D = N*sizeLQ
#     elif i==4:
#         theta_D = theta_base*lightLQ
#         S_D = N*sizeHQ

#     for nD in nD_range:
#         ns = nD+1
#         mu = 0.002 / ns  # Probability of finding shelter
        


#         nLightDistractor = nD*proportion//4



#         def one_distractor_model(t, xyz):
#             X, YD = xyz
#             U = N-X-YD
#             QT = theta_T / (1.0 + rho * ( (X / S_T)**n ))
#             QD = theta_D / (1.0 + rho * ( ((YD / nD) / S_D)**n ))
#             dx = -X * QT + mu * (1.0 - X / S_T) * U
#             dy = -YD * QD + (nD * mu) * (1.0 - YD / (nD * S_D)) * U
#             return np.array([dx, dy])


#         def two_distractor_model(t, xyz):
#             X, YL, YB = xyz
#             U = N - X - YL - YB

#             # Leaving terms (with density effects)
#             QT = theta_T / (1.0 + rho * (X / S_T) ** n)
#             QL = theta_L / (1.0 + rho * (4.0 * YL / ((4-proportion) * (ns - 1) * S_L)) ** n)
#             QB = theta_B / (1.0 + rho * (4.0 * YB / (proportion * (ns - 1) * S_B)) ** n)

#             alpha_T  = mu
#             alpha_DL = mu * (ns - 1) * (4-proportion) / 4
#             alpha_DB = mu * (ns - 1) * proportion/ 4

#             dX  = -X  * QT + alpha_T  * (1.0 - X  / S_T) * U
#             dYL = -YL * QL + alpha_DL * (1.0 - 4.0 * YL / ((4-proportion) * (ns - 1) * S_L)) * U
#             dYB = -YB * QB + alpha_DB * (1.0 - 4.0 * YB / (proportion * (ns - 1) * S_B)) * U
#             return np.array([dX, dYL, dYB])

#         times = np.arange(0.0, max_time, dt)
        
#         if i==0 or i == 4:
#             x0 = np.array([0.0, 0.0])   # start uncommitted
#             sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
#         else:
#             x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
#             sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

#         max_val = np.max(sol.y[0])
#         max_idx = np.argmax(sol.y[0])

#         perf_val.append(max_val/N)

#         if max_val/N > 0.5:
#             # Find the time at which the solution reaches 95% of its maximum
#             time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)
#             all_perf_times[i, nD-1] = time_to_95
#     all_perf_vals.append(perf_val)
# all_perf_times = np.where(all_perf_times==0, np.nan, all_perf_times)    
# fig = plt.figure(figsize = (10,6))
# plt.imshow(all_perf_times, vmin = 20000, vmax = 50000)
# plt.xlabel("Number of distractors")
# plt.xticks([1, 3, 5, 7, 9, 11], [2, 4, 6,8,10, 12])
# plt.ylabel("Ratio of \nlarge:bright distractor")
# plt.yticks([0, 1, 2, 3, 4], ["1:0", "3:1", "1:1", "1:3", "0:1"])
# # plt.legend()
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
# plt.show()

