import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm


distractor_type = "low_light"

sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0


N=100
nDmax=30
theta_base=0.01
rho=1667
n=2
max_time=100000
dt = 0.1


times = np.arange(0, max_time, dt)



# nD_range = np.arange(2, nDmax, 2)
nD_range = np.arange(1, nDmax)
difficulty = np.linspace(1.25, 5, 50)


all_perf_vals = []
all_perf_times = []
n_parallels = []
gradients_series = []
collapse_points = []
for i in tqdm(difficulty):
    if distractor_type == "low_size":
        sizeLQ = float(i)
    else:
        lightLQ = float(i)
    configs = {
    "low_size": [sizeLQ, sizeLQ, lightHQ, lightHQ],
    "low_light": [sizeHQ, sizeHQ, lightLQ, lightLQ],
    "low_size_light": [sizeLQ, sizeLQ, lightLQ, lightLQ],
    "half_size_light": [sizeHQ, sizeLQ, lightLQ, lightHQ],
    }

    config = configs[distractor_type]



    perf_val = []
    perf_time = []

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
        if distractor_type != "half_size_light":
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
        
        
        if distractor_type!="half_size_light":
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
        if t_max_val/N > 0.5:
            # Find the time at which the solution reaches 95% of its maximum
            time_to_95 = next((t for t, y in zip(times, sol.y[0, :]) if y >= 0.95 * t_max_val), None)
            perf_time.append(time_to_95)
        else:
            perf_time.append(None)
    
 

    vals = [float(i) for i in perf_val]
    collapse_point = nD_range[next((i for i, x in enumerate(vals) if x < 0.5), None)]
    b = [i for i in perf_time if type(i)==np.float64]
    n_parallel = [1 for i in range(1, len(b)) if float(b[i])<=float(b[i-1])]
    series = [b[i-1] for i in range(1, len(b)) if float(b[i])>float(b[i-1])]+[b[-1]]
    # series = series[1:-1]
    gradient_series = np.diff(series).mean()
    n_parallels.append(np.sum(n_parallel))
    gradients_series.append(float(gradient_series))
    collapse_points.append(collapse_point)




# plt.rcParams.update({'font.size': 24})

# plt.figure(figsize = (6,4))
# plt.plot(difficulty, n_parallels, linewidth = 4, c='black')
# plt.xlabel("Contrast")
# plt.xticks([2, 3, 4, 5])
# plt.ylabel("Parallel regime size")
# plt.show()


# plt.figure(figsize = (6,4))
# plt.plot(difficulty, gradients_series, linewidth = 4, c='black')
# plt.xlabel("Contrast")
# plt.xticks([2, 3, 4, 5])
# plt.ylabel("Series regime gradient")
# hours = np.array([0, 1])
# seconds = hours*3600
# plt.yticks(seconds, hours)
# plt.show()




plt.rcParams.update({'font.size': 32})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 17), sharex=True)


ax1.plot(difficulty, collapse_points, linewidth = 4, c='black')
ax1.set_ylabel(r"$N_\text{collapse}$")
ax1.set_yticks([11, 12, 13, 14, 15])
ax2.plot(difficulty, n_parallels, linewidth=4, c='black')
ax2.set_ylabel(r"$|\text{Regime}_1|$")

ax3.plot(difficulty, gradients_series, linewidth=4, c='black')
if distractor_type == "low_size":
    ax3.set_xlabel(r"$\phi_D$")
else:
    ax3.set_xlabel(r"$\theta_D$")
ax3.set_ylabel(r"$\frac{\Delta T_{95}}{|\text{Regime}_2|}$")

hours = np.array([0, 1, 2])
seconds = hours * 3600
ax3.set_yticks(seconds)
ax3.set_yticklabels(hours)

plt.xticks([1, 2, 3, 4, 5])

for ax in (ax1, ax2, ax3):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        plt.tight_layout()
plt.show()
