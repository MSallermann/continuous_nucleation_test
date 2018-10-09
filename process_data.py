import numpy as np 


data = np.loadtxt("output_10.txt", delimiter=',', comments='#')
vorticity = data[:, 3]
h_reduced = data[:, 1]

with open("new_data.txt", "w") as f:
    for i in range(len(vorticity)):
        f.write("{:^20.10f} {:^20.10f} {:^20.10f}\n".format(h_reduced[i], vorticity[i], vorticity[i]**2))