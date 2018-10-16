import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt("data_30.txt", comments='#')
h = data[:, 0]
vorticity = data[:, 3]
vorticity2 = data[:, 4]

fig, ax = plt.subplots(1)
ax.plot(h, vorticity, marker = "o",  markersize=2, label = "Vorticity")
ax.plot(h, vorticity2, marker = "o", markersize=2, label = "Vorticity$^2$")
ax.set_xlabel("h")
ax.legend()
fig.savefig("plot_vorticity_30.png", dpi=300, bbox_inches="tight")