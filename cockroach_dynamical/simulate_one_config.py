import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


distractor_type = "low_size"
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
N=1
nDmax=18
theta_base=0.01
rho=1667
n=2
max_time=100000
dt = 0.1


times = np.arange(0, max_time, dt)


leaving_rates = []
maxs = []


# nD_range = np.arange(2, nDmax, 2)
nD_range = np.arange(1, nDmax)


perf_val = []
perf_time = []
perf_ratio = []
for nD in nD_range:
    nS = nD+1
    mu = 0.002 / (1 + nD)  # Probability of finding shelter
    theta_T = theta_base*lightHQ
    theta_L = theta_base*lightHQ
    theta_B = theta_base*lightLQ
    theta_D = theta_base*config[2]

    S_T = N*sizeHQ
    S_L = N*sizeLQ
    S_B = N*sizeHQ
    S_D = N*config[0]
    
    ns = nD+1


    if distractor_type != "half_size_light":
        s = np.concatenate([[sizeHQ * N], np.full(nD, config[0] * N)]) # Shelter capacities
        theta = np.concatenate([[lightHQ * theta_base], np.full(nD, config[2] * theta_base)]) # Shelter light levels
    else: 
        s = np.concatenate([[sizeHQ * N], np.full(nD // 2, config[0] * N), np.full(nD // 2, config[1] * N)]) # Shelter capacities
        theta = np.concatenate([[lightHQ * theta_base], np.full(nD // 2, config[2] * theta_base), np.full(nD // 2, config[3] * theta_base)]) # Shelter light levels




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
    
    
    if distractor_type != "half_size_light":
        x0 = np.array([0.0, 0.0])   # start uncommitted
        sol = solve_ivp(one_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)
    else:
        x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
        sol = solve_ivp(two_distractor_model, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)

    t_max_val = sol.y[0].max()
    t_max_idx = sol.y[0].argmax()
    d_max_val = np.sort(sol.y[1:, -1])[::-1][0]
    
    if distractor_type == "half_size_light":
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


    distractor = sol.y[1]/nD
    leaving_rate = theta_D*distractor/(1+rho*(distractor/S_D)**2)
    leaving_rates.append(leaving_rate[np.argmax(sol.y[1])])

    # plt.plot(sol.y[1])
    # plt.plot(sol.y[0], c='black')
    # plt.scatter(np.argmax(sol.y[1]), sol.y[1,np.argmax(sol.y[1])])
    # plt.show()
    # plt.plot(leaving_rate[1, :max_time]/nD)
    # plt.scatter(np.argmax(sol.y[1]), leaving_rate[1,np.argmax(sol.y[1])]/nD)
    # plt.show()
    # print(np.max(sol.y[1, np.argmax(sol.y[1])])/nD)
    # if nD == 2:
    #     break

# Set global font size
plt.rcParams.update({'font.size': 24})

colors = ['tab:orange', 'tab:red','dodgerblue']
fig=plt.figure(figsize = (6,5))

plt.plot(nD_range, perf_val, linewidth = 4, c='black')

# plt.xlabel("Number of distractors")
# plt.ylabel("Final proportion under target")
plt.ylabel(r"$\bar{X}$")
plt.xticks([0, 5, 10, 15], [])
plt.xlim([0, 18])
plt.ylim([-0.1, 1.1])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.savefig("../figs/mfm_accuracy_with_markers.png", bbox_inches='tight')
plt.show()


serial_prediction = 1800*np.arange(1, 20)+perf_time[0]
exponential_prediction= 1800*np.arange(1, 20)**1.3+ perf_time[0]


fig=plt.figure(figsize = (6,5))
# hours = np.array([0, 2, 4, 6, 8, 10, 12])
hours = np.array([0, 4, 8, 12, 16])
seconds = hours*3600


plt.plot(nD_range, perf_time, linewidth = 4, c='black')
plt.xlabel("Number of distractors")
# plt.ylabel(r"$T_{95}$ decision time (hrs)")
# plt.xticks([])
# plt.xticks(np.arange(1, 14), np.arange(2, 15))
# plt.scatter(nD_range[0], perf_time[0], marker = '*', c='black', s=800, zorder = 5)
# plt.scatter(nD_range[4], perf_time[4], marker = '^', c='black', s=800, zorder = 5)
plt.xlim([0, 13])
plt.yticks(seconds, hours)
plt.yticks(seconds, [])
# plt.axhline(10*3600, xmin = 0.08, c='tab:blue', linestyle = '--', linewidth = 4)
# plt.plot(np.arange(1, 20), serial_prediction, linestyle = '--', c='tab:orange', linewidth = 4)
# plt.plot(np.arange(1, 20), exponential_prediction, linestyle = '--', c='tab:red', linewidth = 4)
plt.ylim([0, 20*3600])
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.savefig("../figs/mfm_t95_lit_rev_preds.png", bbox_inches='tight')
plt.show()
