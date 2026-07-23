import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import *

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

sizeHQ = 1.0
lightHQ = 1.0
N=1
theta_base=0.01
rho=1667
n=2
max_time=1000000
dt = 0.1

nD = 1
times = np.arange(0, max_time, dt)


mu = 0.002/(1+nD)

grid = np.linspace(0, N, 801)
X, Y = np.meshgrid(grid, grid)


def get_ports(sizeLQ, lightLQ):






    s = np.array([N * sizeLQ, N])
    theta = np.array([theta_base * lightLQ, theta_base * 2])

        
    params = {'s': s, 'theta': theta, 'mu': mu, 'rho': rho, 'n': n, 'N': N}

    # Solve the ODE system
    def model(t, x): return ode_sys(t, x, params)
    x0 = np.zeros(len(s))  # Initial conditions
    sol = solve_ivp(model, t_span=(0, max_time), y0=x0, t_eval=times)


    # Parameters

    ns  = nD + 1
    st  = s[0]                  # target capacity
    sd  = s[1]                  # per-distractor capacity (identical distractors)
    theta_t = theta[0]
    theta_d = theta[1]

    def rates(x, y):
        QT = theta_t / (1.0 + rho * ( (x / st)**n ))
        QD = theta_d / (1.0 + rho * ( (y / sd)**n ))
        dx = -x * QT + mu * (1.0 - x / st) * (N - x - y)
        dy = -y * QD + mu * (1.0 - y / sd) * (N - x - y)
        return dx, dy

    
    feasible = (X >= 0) & (Y >= 0) & (X + Y <= N)

    DX, DY = rates(X, Y)
    DX = np.where(feasible, DX, np.nan)
    DY = np.where(feasible, DY, np.nan)

    return DX, DY, sol

sizeLQ = [1.3, 1.6, 1.3]
lightLQ = [1.3, 1.3, 1.6]

plt.rcParams.update({'font.size': 24})






fig, axes = plt.subplots(3, 1, figsize=(9,13), sharex = True)
for i, ax in enumerate(axes):
    DX, DY, sol = get_ports(sizeLQ[i], lightLQ[i])

    # Feasible triangle background 
    ax.fill_between([0, N], [N, 0], color='#99CC66', zorder=0)
    ax.fill_between([0, 0.5], [0.5, 0.5], color = '#F3F4F6', zorder = 0)
    ax.fill_between([0.5, N], [0.5, 0], color = '#FF99CC', zorder = 0)
    
    # Nullclines
    c1 = ax.contour(X, Y, DX, levels=[0], colors='tab:orange', linewidths=4, zorder = 3)
    c2 = ax.contour(X, Y, DY, levels=[0], colors='tab:red',  linewidths=4, zorder =3)


    Um = np.ma.masked_invalid(DX)
    Vm = np.ma.masked_invalid(DY)
    ax.streamplot(grid, grid, Um, Vm, density=0.8, arrowsize=1.2, minlength=0.05, color = 'grey')


    x_traj = sol.y[0]
    y_traj = sol.y[1]

    ax.plot(x_traj, y_traj, color='tab:blue', lw=4, label='simulation', zorder = 4)
    # ax.axhline(0.5, xmax = 0.5, linestyle = '--', c='black', linewidth = 2)
    # ax.axvline(0.5, ymax = 0.5, linestyle = '--', c='black', linewidth = 2)
    ax.set_yticks([0, 0.5, 1],[0, 0.5, 1])
    ax.set_xlim(0, N)
    ax.set_ylim(-0.05, N)
    
    # ax.set_ylabel('Proportion under \nbright shelter')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
axes[2].set_xticks([0, 0.5, 1],[0, 0.5, 1])
# axes[2].set_xlabel('Proportion under \nlarge shelter')
plt.show()






