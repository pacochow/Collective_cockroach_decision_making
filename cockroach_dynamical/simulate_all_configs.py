import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sizeLQ = 5
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0
config1 = [sizeLQ, sizeLQ, lightHQ, lightHQ] 
config2 = [sizeHQ, sizeHQ, lightLQ, lightLQ]
config3 = [sizeLQ, sizeHQ, lightHQ, lightLQ]
N=100
nDmax=30
theta_base=0.01
rho=1667
n=2
max_time=150000
dt = 0.1


times = np.arange(0, max_time, dt)


all_perf_vals = []
all_perf_times = []
all_perf_ratios = []
configs= [config1, config2, config3] 
labels = ["Low size quality", "Low light quality", "1/2 low size and 1/2 low light quality"]
# nD_range = np.arange(2, nDmax, 2)
nD_range = np.arange(1, nDmax)

for i, config in enumerate(configs):
    perf_val = []
    perf_time = []
    perf_ratio = []
    for nD in nD_range:
        mu = 0.002 / (1 + nD)  # Probability of finding shelter
        theta_T = theta_base
        theta_L = theta_base*lightHQ
        theta_B = theta_base*lightLQ
        theta_D = theta_base*config[2]

        S_T = N
        S_L = N*sizeLQ
        S_B = N*sizeHQ
        S_D = N*config[0]
        
        ns = nD+1
        if i != 2:
            s = np.concatenate([[N], np.full(nD, config[0] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
        else: 
            s = np.concatenate([[N], np.full(nD // 2, config[0] * N), np.full(nD // 2, config[1] * N)]) # Shelter capacities
            theta = np.concatenate([[theta_base], np.full(nD // 2, config[2] * theta_base), np.full(nD // 2, config[3] * theta_base)]) # Shelter light levels
        


        # Pack parameters into a dictionary
        params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}
        

        def one_distractor_model(t, xyz):
            X, YD = xyz
            U = N-X-YD
            QT = theta_T / (1.0 + rho * ( (X / S_T)**n ))
            QD = theta_D / (1.0 + rho * ( ((YD / nD) / S_D)**n ))
            dx = -X * QT + mu       * (1.0 - X / S_T)       * U
            dy = -YD * QD + (nD * mu) * (1.0 - YD / (nD * S_D)) * U
            return np.array([dx, dy])


        def two_distractor_model(t, xyz):
            X, YL, YB = xyz
            U = N - X - YL - YB

            # Leaving terms (with density effects)
            QT = theta_T / (1.0 + rho * (X / S_T) ** n)
            QL = theta_L / (1.0 + rho * (2.0 * YL / ((ns - 1) * S_L)) ** n)
            QB = theta_B / (1.0 + rho * (2.0 * YB / ((ns - 1) * S_B)) ** n)

            # Arrival splits: target gets 1/ns, each distractor group gets (ns-1)/(2ns)
            alpha_T  = mu
            alpha_DL = mu * (ns - 1) / 2
            alpha_DB = mu * (ns - 1) / 2

            dX  = -X  * QT + alpha_T  * (1.0 - X  / S_T) * U
            dYL = -YL * QL + alpha_DL * (1.0 - 2.0 * YL / ((ns - 1) * S_L)) * U
            dYB = -YB * QB + alpha_DB * (1.0 - 2.0 * YB / ((ns - 1) * S_B)) * U
            return np.array([dX, dYL, dYB])

        times = np.arange(0.0, max_time, dt)
        
        
        if i!=2:
            x0 = np.array([0.0, 0.0])   # start uncommitted
            sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
        else:
            x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
            sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

        t_max_val = sol.y[0].max()
        t_max_idx = sol.y[0].argmax()
        d_max_val = np.sort(sol.y[1:, -1])[::-1][0]
        
        if i == 2:
            d_max_val/=(nD/2)
        else:
            d_max_val/=nD
        perf_val.append(t_max_val/params['N'])
        perf_ratio.append(t_max_val/d_max_val)
        if t_max_val/N > 0.5:
            # Find the time at which the solution reaches 95% of its maximum
            time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * t_max_val), None)
            perf_time.append(time_to_95)
        else:
            perf_time.append(None)
    all_perf_vals.append(perf_val)
    all_perf_times.append(perf_time)
    all_perf_ratios.append(perf_ratio)



# Set global font size
plt.rcParams.update({'font.size': 24})

colors = ['tab:orange', 'tab:red','dodgerblue']
fig=plt.figure(figsize = (10,5))
for i in range(len(configs)):
    if i == 2:
        plt.plot(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c=colors[i], linestyle = '--')
    else:
        plt.plot(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c=colors[i])
    # plt.scatter(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c='black', zorder = 2)
    
plt.scatter(2, all_perf_vals[-1][0], marker = '*', s = 200, c='black', zorder=2)
plt.scatter(6, all_perf_vals[-1][2], marker = '^', s = 200, c='black', zorder=2)
plt.scatter(12, all_perf_vals[-1][5], marker = 'P', s = 200, c='black', zorder=2)
# plt.xlabel("Number of distractors")
# plt.ylabel("Final proportion under target")
plt.ylabel(r"$\bar{X}$")
plt.xticks([0, 5, 10, 15], [])
plt.xlim([0, 18])
plt.ylim([-0.1, 1.1])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.savefig("../figs/mfm_accuracy_with_markers.png", bbox_inches='tight')
plt.show()

fig=plt.figure(figsize = (10,6))
for i in range(len(configs)):
    if i == 2:
        plt.plot(nD_range, all_perf_ratios[i], linewidth = 4, label = labels[i], c=colors[i], linestyle = '--')
    else:
        plt.plot(nD_range, all_perf_ratios[i], linewidth = 4, label = labels[i], c=colors[i])
# plt.scatter(2, all_perf_vals[-1][0], marker = '*', s = 200, c='black', zorder=2)
# plt.scatter(6, all_perf_vals[-1][2], marker = '^', s = 200, c='black', zorder=2)
# plt.scatter(12, all_perf_vals[-1][5], marker = 'P', s = 200, c='black', zorder=2)
plt.xlabel("Number of distractors")
# plt.ylabel("Final proportion under target")
plt.ylabel(r"$\frac{\bar{X}_T}{\bar{X}_D}$")
plt.xticks([0, 5, 10, 15])
plt.xlim([0, 18])
# plt.legend()
# plt.ylim([-0.1, 1.1])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()





fig=plt.figure(figsize = (10,5))
# hours = np.array([0, 2, 4, 6, 8, 10, 12])
hours = np.array([0, 4, 8, 12, 16])
seconds = hours*3600
experimental_xs = [2, 4, 8]
experimental_ys = [23000, 19000, 29000]

for i in range(len(configs)):
    plt.plot(nD_range, all_perf_times[i], linewidth = 4, label = labels[i], c=colors[i])
    # plt.scatter(nD_range, all_perf_times[i], label = labels[i], linewidth = 4, c='black', zorder = 2)
plt.scatter(2, all_perf_times[-1][0], marker = '*', s = 200, c='black', zorder=2)
plt.scatter(6, all_perf_times[-1][2], marker = '^', s = 200, c='black', zorder=2)
plt.scatter(12, all_perf_times[-1][5], marker = 'P', s = 200, c='black', zorder=2)
# plt.errorbar(experimental_xs, experimental_ys, yerr = 2000, fmt='o', capsize=5, c='black', markersize=12)
plt.xlabel("Number of distractors")
plt.ylabel("$T_{95}$ (hrs)")
plt.xticks([0, 5, 10, 15])
plt.xlim([0, 18])
plt.yticks(seconds, hours)
plt.ylim([0, 65000])

ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.savefig("../figs/mfm_t95_with_markers.png", bbox_inches='tight')
plt.show()




### FOR PRESENTATIONS

# Set global font size
plt.rcParams.update({'font.size': 24})

# hypothetical_ys = all_perf_vals[0][0]**(nD_range)
colors = ['tab:orange', 'tab:red','dodgerblue']
fig=plt.figure(figsize = (12,8))
for i in range(1):
    if i == 2:
        plt.plot(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c=colors[i], linestyle = '--')
    else:
        plt.plot(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c='black')
    plt.scatter(nD_range, all_perf_vals[i], linewidth = 4, label = labels[i], c='black', zorder = 2)
# plt.plot(nD_range, hypothetical_ys, c='black', linestyle = '--', linewidth = 4)
plt.xlabel("Number of distractors")
plt.ylabel("Final proportion under target")
plt.xticks([0, 5, 10, 15])
plt.xlim([0, 18])
plt.ylim([-0.03, 1.03])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()




fig=plt.figure(figsize = (12,8))
hours = np.array([0, 4, 8, 12, 16])
seconds = hours*3600
experimental_xs = [2, 4, 8]
experimental_ys = [23000, 19000, 29000]

for i in range(1):
    plt.plot(nD_range, all_perf_times[i], linewidth = 4, label = labels[i], c='black')
    plt.scatter(nD_range, all_perf_times[i], label = labels[i], linewidth = 4, c='black', zorder = 2)

plt.xlabel("Number of distractors")
plt.ylabel(r"$T_{95}$ (hrs)")
plt.xticks([0, 5, 10, 15])
plt.xlim([0, 18])
plt.yticks(seconds, hours)
plt.ylim([0, 65000])

ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()





## HYPOTHETICAL FIG
fig=plt.figure(figsize = (12,8))
hours = np.array([0, 4, 8, 12, 16])
seconds = hours*3600
experimental_xs = [2, 4, 8]
experimental_ys = [23000, 19000, 29000]
ys = 10000*np.exp(0.1*nD_range[:11])+23816
plt.plot(nD_range[:11], ys, linewidth = 4, linestyle = '--', label = labels[i], c='black')

plt.xlabel("Number of distractors")
plt.ylabel("Decision time (hrs)")
plt.xticks([0, 5, 10])
plt.xlim([0, 13])
plt.yticks(seconds, hours)
plt.ylim([0, 65000])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()





# fig = plt.figure(figsize = (14,6))
# all_perf_vals = []
# all_perf_times = []
# labels = ["Low size quality", "3/4 low size + 1/4 low light quality", "1/2 low size + 1/2 low light quality", "1/4 low size + 3/4 low light quality", "Low light quality"]
# colors = ['tab:orange', 'lightsteelblue', 'dodgerblue', 'navy', 'tab:red']

# for i, proportion in enumerate(np.arange(5)):
#     perf_val = []
#     perf_time = []
#     if proportion == 0 or proportion == 2 or proportion == 4:
#         nD_range = np.arange(2, nDmax, 2)
#     else:
#         nD_range = np.arange(4, nDmax, 4)
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

#         perf_val.append(max_val/params['N'])
#         if max_val/N > 0.5:
#             # Find the time at which the solution reaches 95% of its maximum
#             time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * max_val), None)

#             perf_time.append(time_to_95)
#         else:
#             perf_time.append(np.nan)
#     plt.plot(nD_range, perf_time, label = labels[i], color = colors[i], linewidth = 4)
#         # if i > 0 and i < 4 and nD == 12:
#         #     plt.plot(times, sol.y[0])
#         #     plt.plot(times, sol.y[1])
#         #     plt.plot(times, sol.y[2])
#         #     plt.axvline(time_to_95, c='black', linewidth = 4)
#         #     plt.show()
#     all_perf_vals.append(perf_val)
#     all_perf_times.append(perf_time)


# plt.xlabel("Number of distractors")
# plt.ylabel("$T_{95}$ (s)")
# # plt.legend()
# ax=fig.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(2)
# plt.show()


