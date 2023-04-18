import scipy.integrate
from scipy import stats
import numpy as np

def chargeDist(trj, charge, last_n_frame, binwidth, direction):
    """Make charge density profile (e/nm^3) along one direction 
    Args:
        trj (mdtraj): _description_
        charge (numpy): all atom charge info for all frames, shape = (n_frame, n_atoms)
        last_n_frame (int): _description_
        binwidth (float): nm
        direction (int): 0 is x, 1 is y, 2 is z
    Return:
        charge_den_profile (numpy): charge density profile in one direction
    """
    trj = trj[-last_n_frame:]
    total_xyz = trj.xyz
    total_xyz = np.reshape(total_xyz, (-1,3))
    
    # only focus on the z direcition
    trj_data = total_xyz[:,direction]
    
    # find corresponding charge info
    charge_sliced = charge[-last_n_frame:]
    charge_data = np.reshape(charge_sliced, (1,-1))[0]

    box = trj.unitcell_lengths[0]

    # The last frame xyz
    gro_xyz = trj[-1].xyz[0]
    gro_xyz = gro_xyz[:,direction]
    
    # Determine bins along the direction
    bins = np.arange(min(gro_xyz), max(gro_xyz) + binwidth, binwidth)

    # Charge distribution along the direciton
    charge_distribution = stats.binned_statistic(trj_data, charge_data, 'sum', bins=bins)
    
    bin_volumn = binwidth * box[0] * box[1]
    total_bin_volumn = bin_volumn * last_n_frame

    # averaged charge density along the direction
    avg_charge_density = charge_distribution.statistic/total_bin_volumn
    
    return bins, avg_charge_density

def integrate_poisson_1D(r, charge_dens_profile, periodic=False):
    """Integrate Poisson's equation twice to get electrostatic potential from charge density.

    Inputs:
    r : 1D numpy array. Values of coordinates, in Angstroms.
    charge_dens_profile : 1D numpy array. Values of charge density, 
        in e*nm^-3.
    periodic : Boolean. Optional keyword. If True, adds linear function to 
        ensure values at boundaries are periodic.
    Outputs:
    phi : Numpy array of same length as r. Electrostatic potential in V. 
        The value of phi at the first r value is set to be 0.
    """
    factor0 = 8.8541878128 * 10**(-12) ## factor0 * F/m = vaccume permittivity
    factor1 =  1 # from F to C/V
    factor2 =  6.241506363094 * 10**(18) # from C to e
    factor3 =  10**(-9) # from 1/m to 1/nm
    eps_0_factor = factor0 * factor1 * factor2 * factor3 # from factor0 * F/m to eps_0_factor * e/(V*nm)
    # eps_0_factor = 8.854e-12/1.602e-19*1e-9 # Note: be careful about units!
    int1 = scipy.integrate.cumtrapz(charge_dens_profile, r, initial=0)
    int2 = -scipy.integrate.cumtrapz(int1, r, initial=0)/eps_0_factor
    if periodic:
        # Ensure periodic boundary conditions by adding a linear function such
        # that the values at the boundaries are equal
        phi = int2 - ((int2[-1]-int2[0])/(r[-1]-r[0])*(r-r[0]) + int2[0])
    else:
        phi = int2
    return phi
