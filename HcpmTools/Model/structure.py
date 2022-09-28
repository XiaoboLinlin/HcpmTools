
import numpy as np
def remove_partial_structure(structure, limit, operator):
    """ remove the part of cdc which is below or above limit in axis direction, only for z direction
    Parameters
    --------
    structure: parmed.structure
    limit: float
        below this limit, atoms in this structure will be removed
    """
    axis = 2
    if operator == "<": ## below limit is removed
        print('yes')
        count = np.where(structure.coordinates[:, axis] < limit)
        n = len(count[0])
        for i in range(n):
            for atom in structure.atoms:
                if atom.xz < limit:
                    structure.atoms.remove(atom)
    if operator == ">": ## above limit is removed
        count = np.where(structure.coordinates[:, axis] > limit)
        n = len(count[0])
        for i in range(n):
            for atom in structure.atoms:
                if atom.xz > limit:
                    structure.atoms.remove(atom)
    return structure