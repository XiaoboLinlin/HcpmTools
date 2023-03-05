import os
import numpy as np
import mdtraj as md
from scipy import stats

class Charge_analysis:
    """Analyze the charge induced on the electrodes
    
    Parameters
    ---------
    charge_file (str): The path to the npy file for charge infomation
    gro_file (str): The path to the gro file
    """
    
    def __init__(
        self,
        charge_file,
        gro_file
        ):
        self.charge_file = charge_file
        self.gro_file = gro_file
        self.charge = np.load(charge_file)
        
    def electrode_charge(self, mdtrj_name, timestep = 0.002):
        """Calculate the total electrode charge

        Args:
            mdtrj_name (str, optional): mdtrj select syntax. like 'resname neg' or 'resname pos'
            timestep (float, optional): timestep (ns). Defaults to 0.002.

        Returns:
            time (ns), electrode_charge (e)
        """
        
        gro_frame = md.load(self.gro_file)
        pos_ele = gro_frame.top.select(mdtrj_name)
        pos_charge = self.charge[:,pos_ele]
        sum_pos_q = np.sum(pos_charge, axis = 1)
        xdata = np.arange(0, len(sum_pos_q),1)
        xdata = xdata * timestep
        return xdata, sum_pos_q
    
    def charge_dist(self, mdtrj_name, last_n_frame = 5000, statistic_method = 'mean', binwidth = 0.1):
        """charge per atom distribution in z direction 

        Args:
            mdtrj_name (str, optional): _description_. Defaults to 'resname pos or resname neg'.
            last_n_frame (int, optional): _description_. Defaults to 5000.
            statistic_method (str, optional): _description_. Defaults to 'mean'.
            binwidth (float, optional): _description_. Defaults to 0.1.
        return:
            distance (nm),  charge_dist[0] (e) = distance over z direction, distributed charge (per atom) over electrodes
        """
        
        charge_dist = []
        new_bins = []
        one_frame = md.load(self.gro_file)
        target_top_idx = one_frame.top.select(mdtrj_name)
        target = one_frame.atom_slice(target_top_idx)
        z_data = target.xyz[0,:,2]
        
        ### make averged atom charge over last n frames
        avg_charge = np.mean(self.charge[-last_n_frame:], axis = 0)
        
        ### make bins
        bins = np.arange(0, max(z_data) + binwidth, binwidth)
        ###
        targe_avg_charge = avg_charge[target_top_idx]
        charge_distribution = stats.binned_statistic(z_data, targe_avg_charge, statistic=statistic_method, bins=bins)
        charge_dist.append(charge_distribution.statistic)
        new_bins.append(bins)
        new_bins0= new_bins[0]
        distance = new_bins0[1:] - (new_bins0[1]-new_bins0[0])/2
        return distance, charge_dist[0]
    
    def individual_charge_dist(self, mdtrj_name, atom_num, binwidth = 0.001, range = (-1,1), last_n_frame = 5000, density = True):
        """_summary_

        Args:
            mdtrj_name (str): "pos" or "neg".
            atom_num (int): consider the set of electrode atom types sorted by positions. e.g., when "pos", atom_num = -1 means the H atom on the electrolyte-adjcent layer
            last_n_frame (int, optional): _description_. Defaults to 5000.
            binwidth (float, optional): _description_. Defaults to 0.1.
            range (tuple, optional): _description_. Defaults to (-1,1).
            density (bool): if calculate histogram using probablity density function.
        return counts, bin_edges
        """
        one_frame = md.load(self.gro_file)
        all_z_data = one_frame.xyz[0, :, 2]
        target_top_idx = one_frame.top.select(mdtrj_name)
        target = one_frame.atom_slice(target_top_idx)
        z_data = target.xyz[0,:,2]
        
        
        atom_pos = sorted(list(set(z_data))) # from small value to larger position values
        # type_pos_idx = np.where(atom_pos == atom_pos[atom_num])[0][0]
        # type_pos = atom_pos[type_pos_idx]
        type_pos = atom_pos[atom_num]
        
        frames_charge = self.charge[-last_n_frame:]
        type_pos_one = np.where(all_z_data == [type_pos])[0]
        
        all_frames_charge = frames_charge[:, type_pos_one]
        all_charges = all_frames_charge.flatten()
        
        bins = np.arange(range[0], range[1], binwidth)
        counts, bin_edges = np.histogram(all_charges, bins=bins, density=density)
        
        return counts, bin_edges
        
        
    