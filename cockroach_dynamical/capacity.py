import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0, 100, 0.1)
y = x*0.5/(1+600*(x/100)**2)
y1=x*0.5/(1+600*(x/175)**2)
y2 = x*0.875/(1+600*(x/100)**2)

plt.plot(x, y, linewidth = 4, label = "Target")
plt.plot(x, y1, linewidth = 4, label = "Low size quality")
plt.plot(x, y2, linewidth = 4, label = "Low light quality")
plt.legend()
plt.xlabel("Number of agents under shelter")
plt.ylabel("Leaving term")
plt.show()

diff = []
all_diffs = []
for i, ns in enumerate(np.arange(2, 30)):
    x=np.arange(0, 100/ns, 0.1)
    dx_target = -(0.5*x/(1+600*(x/100)**2))+(1-x/100)*(100-ns*x)/ns
    dx_distract = -(0.5*x/(1+600*(x/175)**2))+(1-x/175)*(100-ns*x)/ns
    all_diffs.append(x[((dx_target-dx_distract)>0).argmax()])

    zero_index = (dx_distract>0).argmin()-1
    diff.append(dx_target[zero_index]-dx_distract[zero_index])
    # print(ns)
    # plt.plot(x, dx_target, linewidth  = 2, label = f"Target, {ns} shelters")
    # plt.plot(x, dx_distract, linewidth  = 2, label = f"Distractor, {ns} shelters", linestyle = '--')
    # plt.show()

plt.scatter(np.arange(2, 30), diff, c='black')
plt.xlabel("Number of shelters")
plt.ylabel("Difference in dX/dt")
plt.axvline(28, c='black', linewidth = 2, linestyle = '--')
plt.ylim([0, 1])
plt.show()

plt.scatter(np.arange(2, 30), all_diffs, c='black')
plt.xlabel("Number of shelters")
plt.ylabel("Number of cockroaches where\n target dx/dt > distractor dx/dt")
plt.axvline(28, c='black', linewidth = 2, linestyle = '--')
plt.show()
# ns = np.arange(1, 100)
# for x_star in [2, 4, 6]:
#     dx = -(0.5*x_star/(1+600*(x_star/100)**2))+(1-x_star/100)*(100-ns*x_star)/ns
#     plt.plot(ns, dx, linewidth  = 2, label = f"$x* = {x_star}$")
# plt.axhline(0, c='black', linewidth = 2)
# plt.legend()
# plt.xlabel("Number of distractors")
# plt.ylabel("$dX/dt$")
# plt.show()

# dx = -0