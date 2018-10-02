path_to_spirit_pkg = "/Users/sallermann/Coding/spirit/core/python"
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


n_trials = 20
n_it = 200000000
size = 20
field = 1.0
J = 2.0
convergence = 0
delta_t = 0.0001
n_iterations = 100000

fields = [field for i in range(n_trials)]

outfile = './output_test_determinism.txt'
image = './images_test_determinism/image.ovf'

import os
if not os.path.exists(os.path.dirname(image)):
    os.makedirs(os.path.dirname(image))

with open(outfile, 'a') as out:
    out.write("#Test reproduciblility")
    out.write("#size = {}, J = {}, convergence = {}, field = {}, delta_t = {}, n_iterations = {}".format(size, J, convergence, field, delta_t, n_iterations))
    out.write("#size, ext field [T], Energy[meV], vorticity\n")

    with state.State("") as p_state:
        parameters.llg.set_output_general(p_state, any=False)
        parameters.llg.set_convergence(p_state, convergence)
        parameters.llg.set_direct_minimization(p_state, False)
        parameters.llg.set_timestep(p_state, delta_t)

        geometry.set_n_cells(p_state, [size, size, size])

        nos = system.get_nos(p_state)
        pos = np.array(geometry.get_positions(p_state)).reshape(nos, 3)

        hamiltonian.set_ddi(p_state, 1)
        hamiltonian.set_exchange(p_state, 1, [J])
        hamiltonian.set_dmi(p_state, 0, [])
        hamiltonian.set_anisotropy(p_state, 0, [0,0,1])
        
        io.image_write(p_state, image)
        
        for i, field in enumerate(fields):
            hamiltonian.set_field(p_state, field, [0,0,1])
            configuration.plus_z(p_state)
            
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = n_iterations)
            io.image_append(p_state, image)

            spins = np.array(system.get_spin_directions(p_state)).reshape(nos, 3)
            Energy = system.get_energy(p_state)
            vorticity = np.abs(getVorticity(spins, size))
                
            out.write("{} {}\n".format(Energy, vorticity))


