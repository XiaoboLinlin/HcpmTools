
import MDAnalysis as mda
import numpy as np
def combineCharge(chargeFile, electrodeId, dataFile):
    """If the charge file only contains the electrode charge information, we use
    this function to combine all atom charge together for the whole systme.

    Args:
        chargeFile (String): numpy .npy file
        electrodeId (list): like [1, 3] (any order is fine)
        dataFile (String): Lammps data file
        
    Return:
        combinedCharge (numpy)
    """
    u = mda.Universe(dataFile)
    charge = np.load(chargeFile)
    electrode = u.select_atoms("resid {} or resid {}".format(electrodeId[0], electrodeId[1]))
    eleid = [atom.id for atom in electrode.atoms]
    
    # Make the eleid starting from 0
    eleid = np.array(eleid) - 1
    
    # all atom charge from data file, round to 7 is fine because we only need fixed charge info of other atoms
    topCharge = [round(atom.charge, 7) for atom in u.atoms]
    topCharge = np.array(topCharge)
    
    combinedCharge = np.tile(topCharge, (np.shape(charge)[0], 1))
    combinedCharge[:, eleid] = charge
    
    return combinedCharge
    