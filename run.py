path_to_spirit_pkg = "/home/moritz/Coding/spirit/core/python"
import sys
sys.path.append(path_to_spirit_pkg)
import time
import datetime
import numpy as np
from spirit import state, simulation, constants, parameters, parameters, geometry, system, hamiltonian, configuration, io

def getVorticity(spins, size):
    result = 0
    for a in range(size):
        result += (np.dot(spins[a + (size-1) * size + size * size * (size-1)], [ -1, 0, 0] ))
        result += (np.dot(spins[a                   + size * size * (size-1)], [1, 0, 0] ))
    for b in range(size):
        result += (np.dot(spins[0      + b * size + size * size * (size-1)], [0, -1, 0]  ))
        result += (np.dot(spins[size-1 + b * size + size * size * (size-1)], [0,  1, 0]  ))
    return result / (4 * size)

def calcReducedField(field):
    return field / (mu_0 * mu_B * 1E30)

def calcJ(size, lam):
    return (size/10.0)**2 * mu_B**2/2 * mu_0 * 1E30

def calcK(Q):
    return mu_B**2 * mu_0 / 2 * 1E30 * 0.001

mu_B = 0.057883817555
mu_0 = 2.0133545E-28

#parameters of experiment
Q = 0
lam = 10

edge_length = 20
size = edge_length
J = calcJ(size, lam)
K = calcK(Q)
convergence = 1E-20
delta_t = 0.001
n_iterations = 100000
n_iterations = 1000


lattice_constant = edge_length / size
mu_s = 1*(lattice_constant)**3

fields = np.arange(1.8, 2, step = 0.01)
fields = [2]


outfile = 'output_{0}.txt'.format(size)
image = './images/image_{}.ovf'.format(size)

import os
if not os.path.exists(os.path.dirname(image)):
    os.makedirs(os.path.dirname(image))

with open(outfile, 'a') as out:
    out.write("#" + str(datetime.datetime.now()) + "\n")
    out.write("#Q                = {}\n".format(Q))
    out.write("#J                = {}\n".format(J))
    out.write("#K                = {}\n".format(K))
    out.write("#edge_length      = {}\n".format(edge_length))
    out.write("#lambda           = {}\n".format(lam))
    out.write("#n_iterations     = {}\n".format(n_iterations))
    out.write("#lattice_constant = {}\n".format(lattice_constant))
    out.write("#mu_s             = {}\n".format(mu_s))
    out.write("#size             = {}\n".format(size))    
    out.write("#field [T], reduced_field, Energy[meV], vorticity\n")

    with state.State("") as p_state:
        parameters.llg.set_output_general(p_state, any=False)
        parameters.llg.set_convergence(p_state, convergence)
        parameters.llg.set_direct_minimization(p_state, False)
        parameters.llg.set_timestep(p_state, delta_t)

        geometry.set_lattice_constant(p_state, lattice_constant)
        geometry.set_n_cells(p_state, [size, size, size])

        nos = system.get_nos(p_state)
        pos = np.array(geometry.get_positions(p_state)).reshape(nos, 3)

        hamiltonian.set_ddi(p_state, 1)
        hamiltonian.set_exchange(p_state, 1, [J])
        hamiltonian.set_dmi(p_state, 0, [])
        hamiltonian.set_anisotropy(p_state, K, [0,0,1])
        
        io.image_write(p_state, image)
        
        for i, field in enumerate(fields):
            hamiltonian.set_field(p_state, field, [0,0,1])
            configuration.plus_z(p_state)
            
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = n_iterations)
            io.image_append(p_state, image)

            spins = np.array(system.get_spin_directions(p_state)).reshape(nos, 3)
            reduced_field = calcReducedField(field)
            Energy = system.get_energy(p_state)
            vorticity = np.abs(getVorticity(spins, size))
                
            out.write("{}, {}, {}, {}\n".format(field, reduced_field, Energy, vorticity))


