"""
Create structures that have larger size for MD, 
the atom charge of the structure is aimed for matching up with charges obtained from DFT
"""
import numpy as np
import parmed as pm


def file_reader(file):
    """read ACF.data file 

    Args:
        file (str): file path
    
    Returns:
        np.array
    """
    with open(file,"r") as fi:
        lines = []
        copy = False
        i = 0
        for ln in fi:
            if ln.startswith(" -"):
                copy = True
                i = i + 1
                continue
            if copy and i == 1:
                lines.append(ln)
    data = np.array(lines)  
    df = np.loadtxt(data) ### don't delete this line (important),used to change ' ...\n' to np.array
    return df


def find_charge(data, want_coor):
    """ find charge for atom with wanted_coordinate

    Args:
        data (np.array): data are from file_reader, and data[:, 1:4] is coordinate
        want_coor: the atom with wanted coordinates
    Returns:
        charge

    """
    data_round = data.round(3)
    want_coor = np.array(want_coor)
    want_coor = want_coor.round(3)
    index = np.where(np.all(data_round[:, 1:4]==want_coor, axis =1))
    # print(want_coor)
    # print(index[0][0])
    charge = data[index[0][0],4]
    return charge

def replicate(structure_sub):
    """set 3*3 replicate for MD
    Args:
        structure_sub (parmed): unit structure
    """
    structure = pm.structure.Structure()
    series = [0,1,2]
    for y in series:
        for x in series:
            structure_sub_copy = structure_sub.__copy__()
            for atom in structure_sub_copy.atoms:
                atom.xx = atom.xx + x * structure_sub.box[0]
                atom.xy = atom.xy + y * structure_sub.box[1]
            structure = structure + structure_sub_copy 
    structure.box = [structure_sub.box[0]*3, structure_sub.box[1] * 3, structure_sub.box[2], 90, 90, 90]
    # structure.box[0] = structure_sub.box[0]*3
    # structure.box[1] = structure_sub.box[1]*3
    # structure.box[2] = structure_sub.box[2]*3
    return structure

