path_to_spirit_pkg = "/Users/sallermann/Coding/spirit/core/python"
import sys
sys.path.append(path_to_spirit_pkg)

import numpy as np
from spirit import state, simulation, constants, parameters, parameters, geometry, system, hamiltonian, configuration, io

size = 5
n_it = 10000
J = 0.084
Q = 0

fields = np.arange(1, 3, step = 0.1)

def getVorticity(spins, size):
    result = 0
    for a in range(size):
        result += (np.dot(spins[a + (size-1) * size + size * size * (size-1)], [ -1, 0, 0] ))
        result += (np.dot(spins[a                   + size * size * (size-1)], [1, 0, 0] ))
    for b in range(size):
        result += (np.dot(spins[0      + b * size + size * size * (size-1)], [0, -1, 0]  ))
        result += (np.dot(spins[size-1 + b * size + size * size * (size-1)], [0,  1, 0]  ))
    return result / (4 * size)


with open('output.txt', 'a') as out:
    out.write("#Q = {0}\n".format(Q))
    out.write("#n_it = {0}\n".format(n_it))
    out.write("#size, ext field, reduced_field, J, vorticity\n")

    with state.State("") as p_state:
        parameters.llg.set_output_general(p_state, any=False)
        parameters.llg.set_convergence(p_state, 0)
        geometry.set_n_cells(p_state, [size, size, size])

        nos = system.get_nos(p_state)
        pos = np.array(geometry.get_positions(p_state)).reshape(nos, 3)

        hamiltonian.set_exchange(p_state, 1, [0.1])
        hamiltonian.set_dmi(p_state, 0, [])
        hamiltonian.set_ddi(p_state, 1)
        
        for field in fields:
            hamiltonian.set_field(p_state, field, [0,0,1])
            configuration.plus_z(p_state)
            simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, n_iterations = n_it)
            spins = np.array(system.get_spin_directions(p_state)).reshape(nos, 3)

            vorticity = np.abs(getVorticity(spins, size))
            print(vorticity)
            out.write("{0}, {1}, {2}\n".format(size, field, vorticity))

            # io.chain_write(p_state, "image" + str(field))


