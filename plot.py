import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt('output.txt', comments='#', delimiter=',')
field     = data[:,1]
vorticity = data[:,2]

plt.plot(field, vorticity, ls="None", marker = "o")
plt.xlabel("External field strength")
plt.ylabel("Vorticity")
plt.savefig('plot.png', dpi=300)

