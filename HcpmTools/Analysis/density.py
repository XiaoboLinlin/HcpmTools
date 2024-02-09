import numpy as np
from scipy import stats
import signac
import os

from sympy import total_degree
project = signac.get_project()

def find_cdc_boundary(one_trj, res_id):
    """the z value for cdc edge boundary

    Args:
        one_frame_trj (_type_): _description_
        res_id (int): if 0, left side cdc edge; if 2, right side cdc edge
        

    Returns:
        z value for boundary (float): unit is nm
    """
    one_frame_trj = one_trj.atom_slice(one_trj.top.select('resid {}'.format(res_id)))
    if res_id == 0:
        max_value = max(one_frame_trj.xyz[0,:,2])
        return max_value
    else:
        min_value = min(one_frame_trj.xyz[0,:,2])
        return min_value
    
def num_density(trj, density=False, axis = 2):
    """calcuate the number of particles or density in the full box range based histrogram

    Args:
        trj (md.traj): can be one frame or multiple frames
        density (bool, optional): _description_. Defaults to False.

    Returns:
        hist: _description_
    """
    total_xyz  = trj.xyz
    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:,axis]
    binwidth = 0.1
    box = trj.unitcell_lengths[0]
    bins = np.arange(0, box[2], binwidth)
    hist, bin_edges = np.histogram(data, bins=bins)
    
    if density:
        bin_volumn = binwidth * box[0] * box[1]
        total_bin_volumn = bin_volumn * len(trj)
        new_hist = hist/total_bin_volumn
        
    new_bins = bins + (bins[1] - bins[0])/2 
    new_bins = new_bins[:-1]
    
    return new_bins, new_hist


def num_density_chunk(trj, bins, axis = 2):
    """calcuate the number of particles or density in the a customized chunk range

    Args:
        trj (md.traj): can be one frame or multiple frames

    Returns:
        bins, hist: _description_
    """
    total_xyz  = trj.xyz
    total_xyz = np.reshape(total_xyz, (-1,3))
    data = total_xyz[:,axis]
    hist, bin_edges = np.histogram(data, bins=bins)
    return bins, hist


def chunk_mean(trj_com, bins, new_time_space = 0.002, avg_frames = False):
    """calculte the average number of molecules in a chunk time, and also in a chunk distance

    Args:
        trj_com (_type_): _description_
        bins (_type_): for distance chunk range
        new_time_space (float): ns
        avg_frames: if to average several frames to one frames
        
    Returns:
        bins, mean_value (distance, time): for example, mean_value[0] is the first bin range of values
    """
    hist_total =[]
    # bins, hist = num_density_chunk(trj_com[0], bins = [0, 5, left_boundary])
    ###calculate each frame histogram using custimized bins
    for i in range(len(trj_com)):
        bins, hist = num_density_chunk(trj_com[i], bins = bins)
        hist_total.append(hist)
    
    hist_total = np.array(hist_total)
    if avg_frames:
        t = np.arange(len(trj_com)) * 0.002 ## 0.002 is timestep
        new_t = np.arange(t[0], t[-1], new_time_space)
        mean_value = []
        
        ### average the valus from several frames to one averaged frame, i determine which bin range
        for i in range(len(bins)-1):
            charge_distribution = stats.binned_statistic(t, hist_total[:,i], 'mean', bins=new_t)
            mean_value.append(charge_distribution.statistic)
        mean_value = np.array(mean_value)
    else:
        mean_value = hist_total.T
    
    return mean_value
