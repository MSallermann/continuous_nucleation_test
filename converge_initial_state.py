path_to_spirit_pkg = "/home/moritz/Coding/spirit/core/python"
# path_to_spirit_pkg = "/Users/sallermann/Coding/spirit/core/python"

import sys
sys.path.append(path_to_spirit_pkg)
import time
import datetime
import numpy as np
from spirit import state, simulation, constants, parameters, parameters, geometry, system, hamiltonian, configuration, io

def calcJ(size, lam):
    return (size/10.0)**2 * mu_B**2/2 * mu_0 * 1E30

def calcK(Q):
    return Q * mu_B**2 * mu_0 / 2 * 1E30 * 0.001

mu_B = constants.mu_B
mu_0 = constants.mu_0

#parameters of experiment
Q = 0.001
lam = 10

edge_length = 20
size = edge_length
J = calcJ(size, lam) * 2
K = 0.000337
convergence = 1E-10
delta_t = 0.0001
n_iterations = 100000

lattice_constant = edge_length / size
mu_s = 1*(lattice_constant)**3

field = 0

image = "./initial_image_{}.ovf".format(size)

import os
if not os.path.exists(os.path.dirname(image)):
    os.makedirs(os.path.dirname(image))

with state.State("") as p_state:
    parameters.llg.set_output_general(p_state, any=False)
    parameters.llg.set_convergence(p_state, convergence)
    parameters.llg.set_direct_minimization(p_state, True)
    parameters.llg.set_timestep(p_state, delta_t)

    geometry.set_lattice_constant(p_state, lattice_constant)
    geometry.set_n_cells(p_state, [size, size, size])

    nos = system.get_nos(p_state)
    pos = np.array(geometry.get_positions(p_state)).reshape(nos, 3)

    hamiltonian.set_ddi(p_state, 1)
    hamiltonian.set_exchange(p_state, 1, [J])
    hamiltonian.set_dmi(p_state, 0, [])
    hamiltonian.set_anisotropy(p_state, K, [0,0,1])
    hamiltonian.set_field(p_state, field, [0,0,1])
    configuration.plus_z(p_state)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = n_iterations)
    io.image_write(p_state, image)


