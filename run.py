path_to_spirit_pkg = "/home/moritz/Coding/spirit/core/python"
import sys
sys.path.append(path_to_spirit_pkg)

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
    return (size * 1E-10)**2 / lam**2 * (mu_0 * (mu_B * 1E30)**2)/2 * 1E-10


mu_B = 0.057883817555
mu_0 = 2.0133545E-28

Q = 0
lam = 10

#edge length of the cube in Angstroem
edge_length = 30

#edge length of cube in number of atoms
size = 30

#number of iterations
n_it = 20000

J = calcJ(size, lam)

lattice_constant = edge_length / size
mu_s = 1*(lattice_constant)**3

# fields = np.arange(1.5, 2.5, step = 0.3)
# outfile = 'output_{0}.txt'.format(size)

fields = [1.5, 1.5, 1.5, 1.5, 1.5]
outfile = 'output_test_determinism.txt'.format(size)
image_folder = './images_test_determinism/'

import os
if not os.path.exists(os.path.dirname(image_folder)):
    os.makedirs(os.path.dirname(image_folder))

with open(outfile, 'a') as out:
    out.write("#" + str(datetime.datetime.now()) + "\n")
    out.write("#Q = {0}\n".format(Q))
    out.write("#J = {0}\n".format(J))
    out.write("#edge_length = {0}\n".format(edge_length))
    out.write("#lambda = {0}\n".format(lam))
    out.write("#n_it = {0}\n".format(n_it))
    out.write("#lattice_constant = {0}\n".format(lattice_constant))
    out.write("#mu_s = {0}\n".format(mu_s))
    out.write("#size, ext field [T], reduced_field, Energy[meV], vorticity\n")

    with state.State("") as p_state:
        parameters.llg.set_output_general(p_state, any=False)
        parameters.llg.set_convergence(p_state, 1E-8)
        parameters.llg.set_direct_minimization(p_state, True)
        # print("before set mu_s")
        # geometry.set_mu_s(p_state, mu_s)
        # print("after set mu_s")
        geometry.set_lattice_constant(p_state, lattice_constant)
        geometry.set_n_cells(p_state, [size, size, size])

        nos = system.get_nos(p_state)
        pos = np.array(geometry.get_positions(p_state)).reshape(nos, 3)

        # hamiltonian.set_anisotropy(p_state, Q*)
        hamiltonian.set_exchange(p_state, 1, [0.1])
        hamiltonian.set_dmi(p_state, 0, [])
        hamiltonian.set_ddi(p_state, 1)
        
        for field in fields:
            hamiltonian.set_field(p_state, field, [0,0,1])
            configuration.plus_z(p_state)
            configuration.add_noise(p_state, 2)
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = int(n_it/3))
            configuration.add_noise(p_state, 1)
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = int(n_it/3))
            configuration.add_noise(p_state, 1)
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = int(n_it/3))

            spins = np.array(system.get_spin_directions(p_state)).reshape(nos, 3)
            reduced_field = calcReducedField(field)
            Energy = system.get_energy(p_state)
            vorticity = np.abs(getVorticity(spins, size))
            io.image_write(p_state, image_folder + "/size_{}_field_{}_{}.ovf".format(size, field, datetime.datetime.now()))
            out.write("{0}, {1}, {2}, {3}, {4}\n".format(size, field, reduced_field, Energy, vorticity))


