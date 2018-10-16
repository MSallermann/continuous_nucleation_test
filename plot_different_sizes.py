import numpy as np 
import matplotlib.pyplot as plt

data_10 = np.loadtxt('output_10.txt', comments='#')
data_20 = np.loadtxt('output_20.txt', comments='#')
data_30 = np.loadtxt('output_30.txt', comments='#')
data_50 = np.loadtxt('output_50.txt', comments='#')
data_100 = np.loadtxt('output_100.txt', comments='#')


h = data_10[:, 1]

vorticity_10 = data_10[:, 3]
vorticity_20 = data_20[:, 3]
vorticity_30 = data_30[:, 3]
vorticity_50 = data_50[:, 3]
vorticity_100 = data_100[:, 3]

vorticity2_10 = data_10[:, 4]
vorticity2_20 = data_20[:, 4]
vorticity2_30 = data_30[:, 4]
vorticity2_50 = data_50[:, 4]
vorticity2_100 = data_100[:, 4]


fig, ax = plt.subplots(1)
ax.plot(h, vorticity_10, marker = "o", markersize = 6, label = "Edge Length = 10 [$\mathring{A}$]")
ax.plot(h, vorticity_20, marker = "o", markersize = 6, label = "Edge Length = 20 [$\mathring{A}$]")
ax.plot(h, vorticity_30, marker = "o", markersize = 6, label = "Edge Length = 30 [$\mathring{A}$]")
ax.plot(h, vorticity_50, marker = "o", markersize = 6, label = "Edge Length = 50 [$\mathring{A}$]")
# ax.plot(h, vorticity_100, marker = "o", markersize = 4, label = "Edge Length = 100 [$\mathring{A}$]")

ax.legend()
ax.set_xlabel("h")
ax.set_ylabel("Vorticity")
fig.savefig('plot_vorticity_vs_edge_length.png', dpi=300, bbox_inches="tight")