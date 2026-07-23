import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


distractor_type = "half_size_light" 
sizeLQ = 1.75
sizeHQ = 1.0
lightLQ = 1.75
lightHQ = 1.0

N = 1.0
theta_base = 0.01
rho = 1667
n = 2
max_time = 100000.0
dt = 0.1

nD = 4                    # total distractors
ns  = nD + 1                 # total shelters (target + distractors)

S_T = sizeHQ * N
S_L = sizeLQ * N          # low-light group has HQ size
S_B = sizeHQ * N          # low-size group has LQ size
theta_T = theta_base
theta_L = lightHQ * theta_base  # low-light => higher leaving
theta_B = lightLQ * theta_base


def rhs3(t, xyz):
    X, YL, YB = xyz
    U = N - X - YL - YB

    # Leaving terms (with density effects)
    QT = theta_T / (1.0 + rho * (X / S_T) ** n)
    QL = theta_L / (1.0 + rho * (2.0 * YL / ((ns - 1) * S_L)) ** n)
    QB = theta_B / (1.0 + rho * (2.0 * YB / ((ns - 1) * S_B)) ** n)

    # Arrival splits: target gets 1/ns, each distractor group gets (ns-1)/(2ns)
    alpha_T  = 0.002/ns
    alpha_DL = 0.002*(ns - 1) / (2*ns)
    alpha_DB = 0.002*(ns - 1) / (2*ns)

    dX  = -X  * QT + alpha_T  * (1.0 - X  / S_T) * U
    dYL = -YL * QL + alpha_DL * (1.0 - 2.0 * YL / ((ns - 1) * S_L)) * U
    dYB = -YB * QB + alpha_DB * (1.0 - 2.0 * YB / ((ns - 1) * S_B)) * U
    return np.array([dX, dYL, dYB])

# -------------- Integrate 3D system --------------
times = np.arange(0.0, max_time, dt)
x0 = np.array([0.0, 0.0, 0.0])   # start uncommitted
sol = solve_ivp(rhs3, (times[0], times[-1]), x0, t_eval=times, rtol=1e-8, atol=1e-10)


# -------------- Phase portrait on slice X_T = X_SLICE --------------
# x_time_index = 20000
# x_time_index = 2000
x_time_index = 120000
# x_time_index = 300000

X_SLICE = sol.y[0, x_time_index]  

# Grid over (Y_L, Y_B)
grid = np.linspace(0.0, N, 801)

# Optional capacity-based caps to avoid singularities at (1 - 2Y/(...)) -> 0
YL_cap = (ns - 1) * S_L / 2.0 * 0.999
YB_cap = (ns - 1) * S_B / 2.0 * 0.999

YLg, YBg = np.meshgrid(grid, grid, indexing="xy")

# Feasible region for this slice: YL >= 0, YB >= 0, X_SLICE >= 0, YL + YB <= N - X_SLICE
feasible = (
    (YLg >= 0) & (YBg >= 0) &
    (YLg + YBg <= N - X_SLICE) &
    (YLg <= YL_cap) & (YBg <= YB_cap) &
    (0 <= X_SLICE <= S_T)  # optional guard
)

def rates_on_Xslice(YL, YB):
    U  = N - X_SLICE - YL - YB
    QL = theta_L / (1.0 + rho * (2.0 * YL / ((ns - 1) * S_L)) ** n)
    QB = theta_B / (1.0 + rho * (2.0 * YB / ((ns - 1) * S_B)) ** n)

    alpha_DL = 0.002*(ns - 1) / (2.0 * ns)
    alpha_DB = 0.002*(ns - 1) / (2.0 * ns)

    dYL = -YL * QL + alpha_DL * (1.0 - 2.0 * YL / ((ns - 1) * S_L)) * U
    dYB = -YB * QB + alpha_DB * (1.0 - 2.0 * YB / ((ns - 1) * S_B)) * U
    return dYL, dYB

DYL, DYB = rates_on_Xslice(YLg, YBg)
DYL = np.where(feasible, DYL, np.nan)
DYB = np.where(feasible, DYB, np.nan)

plt.rcParams.update({'font.size': 24})

fig, ax = plt.subplots(figsize=(6.75, 5))

# Feasible triangle background for the slice (line YL + YB = N - X_SLICE)
ax.fill([0, N - X_SLICE, 0], [0, 0, N - X_SLICE], color='#F3F4F6', zorder=0)

# Nullclines on the slice: dYL = 0 (solid), dYB = 0 (dashed)
ax.contour(YLg, YBg, DYL, levels=[0], colors='tab:orange', linewidths=4, zorder = 5)
ax.contour(YLg, YBg, DYB, levels=[0], colors='tab:red', linewidths=4, zorder = 5)

# Streamplot of the projected field (dYL, dYB)
Um = np.ma.masked_invalid(DYL)
Vm = np.ma.masked_invalid(DYB)
ax.streamplot(grid, grid, Um, Vm, density=1.6, arrowsize=1.2, minlength=0.05, color='silver')

# Project the 3D trajectory onto (Y_L, Y_B)
YL_traj = sol.y[1]
YB_traj = sol.y[2]
# Find and plot the point where X_T is closest to the slice value
X_traj = sol.y[0]
# idx = np.argmin(np.abs(X_traj - X_SLICE))  # time index closest to X_SLICE
idx = x_time_index
ax.plot(YL_traj[idx], YB_traj[idx],
        marker='s', markersize=20, color='yellow', zorder=5, markeredgecolor = 'black')

# (optional) annotate
# ax.annotate(f"t={times[idx]:.1f}",
#             xy=(YL_traj[idx], YB_traj[idx]), xytext=(8, 8),
#             textcoords="offset points")
# ax.legend(loc='upper right')

ax.set_xlim(0, N); ax.set_ylim(0, N)
ax.set_xticks([0, 0.5, 1.0], [0, 0.5, 1.0])
ax.set_yticks([0, 0.5, 1.0], [0, 0.5, 1.0])
# ax.set_xlabel('Proportion under \nlarge distractors')
# ax.set_ylabel('Proportion under \nbright distractors')
ax.set_xlabel("$X_L$")
ax.set_ylabel("$X_B$")
for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_linewidth(2)
plt.tight_layout()
plt.savefig(f"../figs/3D_phase_port_{x_time_index}", bbox_inches = 'tight')
plt.show()


### INSET FIGURE
fig, ax = plt.subplots(figsize=(2, 2))

# Feasible triangle background for the slice (line YL + YB = N - X_SLICE)
ax.fill([0, N - X_SLICE, 0], [0, 0, N - X_SLICE], color='#F3F4F6', zorder=0)

# Nullclines on the slice: dYL = 0 (solid), dYB = 0 (dashed)
ax.contour(YLg, YBg, DYL, levels=[0], colors='black', linewidths=3)
ax.contour(YLg, YBg, DYB, levels=[0], colors='black', linewidths=3)

# Streamplot of the projected field (dYL, dYB)
Um = np.ma.masked_invalid(DYL)
Vm = np.ma.masked_invalid(DYB)
ax.streamplot(grid, grid, Um, Vm, density=0.5, arrowsize=1.6, minlength=0.05)

# Project the 3D trajectory onto (Y_L, Y_B)
YL_traj = sol.y[1]
YB_traj = sol.y[2]
# Find and plot the point where X_T is closest to the slice value
X_traj = sol.y[0]
# idx = np.argmin(np.abs(X_traj - X_SLICE))  # time index closest to X_SLICE
idx = x_time_index
ax.plot(YL_traj[idx], YB_traj[idx],
        marker='o', markersize=15, color='yellow', zorder=5, markeredgecolor = 'black')


ax.set_xlim(0, 0.5); ax.set_ylim(0.2, 0.7)
ax.set_xticks([])
ax.set_yticks([])
for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_linewidth(2)
plt.tight_layout()
plt.show()
