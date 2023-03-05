import numpy as np
from scipy import stats

def calc_msd(traj, timestep = 1, dims=[1, 1, 1], fit_with='scipy'):
    """Calculate the MSD and diffuvisity of a bulk simulation

    Parameters
    ----------
    traj : md.Trajectory
        mdtraj Trajectory

    dims : list of int, default = [1, 1, 1]
        Dimensions to consider, in order of `x, y, z`. A value of 1 will result
        in that dimension being considered.

    fit_with : str, default='scipy.stats'
        Package to use for linear fitting. Options are 'numpy' (`numpy.polyfit`) and
        'scipy' (`scipy.stats.linregress`)

    Returns
    -------
    D : Float
        Bulk 3-D self-diffusvity
    msd : np.ndarray
    """


    if dims == [1, 1, 1]:
        msd = [np.sum((traj.xyz[index, :, :] - traj.xyz[0, :, :]) ** 2) / traj.n_atoms for index in range(traj.n_frames)]
    else:
        msd = np.zeros(shape=traj.n_frames)
        for dim, check in enumerate(dims):
            if check == 1:
                msd += [np.sum((traj.xyz[index, :, dim] - traj.xyz[0, :, dim]) ** 2) / traj.n_atoms for index in range(traj.n_frames)]
            elif check != 0:
                raise ValueError('Indices of dim must be 0 or 1!')

    y = msd
    x = traj.time * timestep - traj.time[0]

    if fit_with == 'scipy':
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        D = slope / (2*np.sum(dims)) * 1e-6

        x_fit = x[int(np.round(len(msd)/10, 0)):]
        y_fit = slope * x_fit + intercept

    elif fit_with == 'numpy':
        fit = np.poly1d(
                np.polyfit(x[int(np.round(len(msd)/10, 0)):],
                           y[int(np.round(len(msd)/10, 0)):],
                           1)
                )

        D = fit[0]/(2*np.sum(dims)) * 1e-6

        x_fit = x[int(np.round(len(msd)/10, 0)):]
        y_fit = fit(x_fit)

    return D, msd, x_fit, y_fit