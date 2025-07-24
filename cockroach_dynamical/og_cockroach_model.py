import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from tqdm import tqdm

def dxdt(t, x, p):
    d = x/p['S']
    q = p['theta']/(1+p['rho']*d**2)
    dx = -x*q+p['mu']*(1-d)*(p['N']-np.sum(x))
    return dx

dt = 0.01
t_max = 10000
t= np.arange(0, t_max, dt)
x0 = [200, 800]



# SIMULATE AND PLOT MODEL
params = {
    'N': 1000,
    'S': [1000, 1100],
    'mu': [0.9, 0.9],
    'theta': 0.5,
    'rho': 600,
}

def model(t, x):
        return dxdt(t, x, params)


sol = solve_ivp(model, t_span = (0, max(t)), y0 = x0, t_eval = t)
plt.plot(t, sol.y[0], linewidth = 3, label = '$X_1$')
plt.plot(t, sol.y[1], linewidth = 3, label = "$X_2$")
plt.legend()
plt.ylim([-50, 1000])
plt.xlabel("Time")
plt.ylabel("Number of individuals under shelter")
plt.show()

# PLOT NULLCLINES
x = np.arange(1, 1000, 0.01)
q1 = params['theta']/(1+params['rho']*(x/params['S'][0])**2)
q2 = params['theta']/(1+params['rho']*(x/params['S'][1])**2)
x2 = params['N'] - x-x*q1/(params['mu'][0]*(1-x/params['S'][0]))
x1 = params['N'] - x-x*q2/(params['mu'][1]*(1-x/params['S'][1]))


plt.scatter(x, x2, s=5, label = '$X_1$ nullcline')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.scatter(x1, x, s=5, label = '$X_2$ nullcline')
plt.scatter(sol.y[0], sol.y[1], s=5, label = 'Trajectory')
plt.xlim([0, 1000])
plt.ylim([0, 1000])
plt.legend()
plt.show()



# FIND FIXED POINTS

# def fixed_point_equations(x):
#     X1, X2 = x
#     q1 = params['theta'] / (1 + params['rho'] * (X1 / params['S'][0])**2)
#     q2 = params['theta'] / (1 + params['rho'] * (X2 / params['S'][1])**2)
#     f1 = -X1 * q1 + params['mu'][0] * (1 - X1 / params['S'][0]) * (params['N'] - X1 - X2)
#     f2 = -X2 * q2 + params['mu'][1] * (1 - X2 / params['S'][1]) * (params['N'] - X1 - X2)
#     return [f1, f2]

# # initial_guesses = [[900, 100]]
# # fixed_points = [fsolve(fixed_point_equations, guess) for guess in initial_guesses]

# # # Remove duplicate fixed points by rounding and using a set
# # fixed_points = np.unique(np.round(fixed_points, 4), axis=0)


# def fixed_point_equations(x):
#     X1, X2 = x
#     q1 = params['theta'] / (1 + params['rho'] * (X1 / params['S'][0])**2)
#     q2 = params['theta'] / (1 + params['rho'] * (X2 / params['S'][1])**2)
#     f1 = -X1 * q1 + params['mu'][0] * (1 - X1 / params['S'][0]) * (params['N'] - X1 - X2)
#     f2 = -X2 * q2 + params['mu'][1] * (1 - X2 / params['S'][1]) * (params['N'] - X1 - X2)
#     return [f1, f2]

# # Generate a grid of initial guesses
# X1_vals = np.linspace(0, params['N'], 5)
# X2_vals = np.linspace(0, params['N'], 5)
# initial_guesses = [[x1, x2] for x1 in X1_vals for x2 in X2_vals]

# # Solve for fixed points
# fixed_points = []
# for guess in initial_guesses:
#     solution, info, success, _ = fsolve(fixed_point_equations, guess, full_output=True)
#     if success:
#         fixed_points.append(solution)

# # Convert to array and remove duplicates using cKDTree clustering
# fixed_points = np.array(fixed_points)
# fixed_points = np.unique(np.round(fixed_points, 4), axis=0)


# plt.scatter(fixed_points[:, 0], fixed_points[:,1], s = 100, c='black', marker = 'X', alpha = 0.5, label = "Fixed point")
# plt.legend()
# plt.show()


# # VARY S2
# x1_final = []
# for i in tqdm(np.arange(1, 1500)):
#     if i<800 or i > 1400:
#         t = 1000
#     elif i < 950 or i > 1050:
#         t = 10000
#     else:
#         t = 100000
#     t= np.arange(0, t_max, dt)
    
#     params = {
        
#         'N': 1000,
#         'S': [1000, i],
#         'mu': [0.9, 0.9],
#         'theta': 0.5,
#         'rho': 600,
#     }

#     def model(t, x):
#         return dxdt(t, x, params)

#     sol = solve_ivp(model, t_span = (0, max(t)), y0 = x0, t_eval = t)
#     x1_final.append(sol.y[0, -1])


# plt.scatter(np.arange(1, 1500)/params['N'], np.array(x1_final)/params['N'], c='black')
# plt.xlabel("Ratio $S_2/S_1$")
# plt.ylabel("Proportion of individuals picking optimal shelter")
# plt.ylim(0, 1)
# plt.axhline(0.5, linestyle = '--', c='black')
# plt.show()





# x10 = 0
# x20 = 0
# x30 = 0
# n = 10
# mu1 = 0.9
# mu2 = 0.9
# theta = 0.5
# p = 600
# s1 = 10
# s2 = 5
# dt = 0.1

# t_max = 10000

# x1 = x10
# x2 = x20
# x3 = x30
# x1s = np.zeros((int(t_max/dt)))
# x1s[0] = 10
# x2s = np.zeros((int(t_max/dt)))
# t= np.arange(0, t_max, dt)

# for t in range(int(t_max/dt)):
#     x1s[t] = x1
#     q1 = theta/(1+p*((x1/s1)**2))
#     dx1 = -x1*q1+mu1*(1-x1/s1)*(n-x1-x2)-np.random.rand()

#     x2s[t] = x2
#     q2 = theta/(1+p*((x2/s2)**2))
#     dx2 = -x2*q2+mu2*(1-x2/s2)*(n-x1-x2)-np.random.rand()

#     # x3s[t] = x3
#     # q3 = theta/(1+p*((x3/s3)**2))
#     # dx3 = -x3*q3+mu3*(1-x3/s3)*(n-x1-x2-x3)

#     x1 += dx1*dt
#     x2 += dx2*dt
#     # x3 += dx3*dt
 
# plt.scatter(np.arange(0, t_max, dt), x1s, label = 'Resource 1')
# plt.scatter(np.arange(0, t_max, dt), x2s, label = 'Resource 2')
# # plt.scatter(np.arange(0, t_max, dt), x3s, label = 'Resource 3')
# plt.xlabel("Time (s)")
# plt.ylabel("Number of individuals")
# plt.legend()
# plt.show()

