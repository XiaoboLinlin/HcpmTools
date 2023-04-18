import numpy as np
import signac
import os
project = signac.get_project()

class integralCapa:
    def __init__(self, eleChargeFileName,
                potentialFileName,
                predict_end = False):
        """_summary_

        Args:
            voltage (int): _description_
            eleChargeFileName: Accumulated charge for total electrode as a function of time
            seed (int): _description_
        """
        self.eleChargeFileName = eleChargeFileName
        self.potentialFileName = potentialFileName
        self.predict_end = predict_end
        
        
    def get_potential(self, case, range, voltage, seed):
        """get potential at different location range

        Args:
            case (str): the case name
            range (str, optional): 'middle' or 'start' or 'end' location. Defaults to 'middle'.
            voltage (int, optional): _description_. Defaults to 0.
            seeds (list, optional): _description_. Defaults to [0,1,2,3].

        Returns:
            float: the calculated values of potential at different locations
        """

        for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
            potential_file = os.path.join(job.workspace(), self.potentialFileName)
            data = np.load(potential_file)
            # data = data["phi"]
            # x = data[:,0]/10  
            y = data["phi"]
            length= len(y)
            # the range is middle 
            if range == 'middle':
                lower_range = int(length/2-length/8)
                upper_range = int(length/2+length/8)
                middle_potential = y[lower_range:upper_range]
                potential_ = np.mean(middle_potential)
            elif range == 'start':
                potential_= y[0]
            elif range == 'end':
                if self.predict_end:
                    potential_ = -voltage
                else:
                    potential_= y[-1]
        return potential_

    def get_electrode_potential_diff(self, case, side, voltage, seed):
        """ get the electrode potential (the difference between electrode and electrolyte) 
            relative to the pzc (the difference between electrode and electrolyte)

        Args:
            side (str): 'positive' means positive electrode
            voltage (float): _description_
        Returns:
            float_: the value of the electrode potential relative to the pzc
        """
        if side == 'positive':
            range_ = 'start'
        elif side == 'negative':
            range_ = 'end' 
        start_po = self.get_potential(case, range= range_, voltage=voltage, seed=seed)
        middle_po = self.get_potential(case, range='middle',voltage=voltage, seed=seed)
        electrode_potential = start_po - middle_po
        start_po_pzc = self.get_potential(case, range= range_, voltage=0, seed=seed)
        middle_po_pzc = self.get_potential(case, range='middle',voltage=0, seed=seed)
        electrode_potential_pzc = start_po_pzc - middle_po_pzc
        electrode_potential_diff = electrode_potential - electrode_potential_pzc
        return electrode_potential_diff 
        
    def get_electrode_charge(self, case, voltage, seed, fraction = 1/5):
        """calcualte the charge accumulated in the positive electrode

        Args:
            case (str): _description_
            voltage (float): _description_
            seed (int): _description_
            fraction (_type_, optional): _description_. Defaults to 1/5.

        Returns:
            float: the avg accumulated charge over the last fraction of charge
        """
        for job in project.find_jobs({"case": case, "voltage": voltage, "seed": seed}):
            charge_file = os.path.join(job.workspace(), self.eleChargeFileName)
            charge = np.load(charge_file)
            charge =charge[:,1]
            length = len(charge)
            charge_fraction = int(length*fraction)
            charge = charge[-charge_fraction:]
            avg_charge = np.mean(charge)
        return avg_charge
        
    def unit_convert(self, charge, voltage, atom_mass =  12.011, n_atom=3620):
        """convert e/(g*V) to F/g

        Args:
            charge (float): the value of elementary charge
            voltage (float):
            atom_mass (float): the mass for one atom, unit is amu
            n_atom (int): the number of atoms

        Returns:
            float: the value of capacitance (F/g)
        """
        import unyt as unit
        voltage = voltage * unit.V
        cdc_mass = atom_mass * n_atom * unit.amu
        charge_value = charge * unit.qp
        capa = charge_value / cdc_mass / voltage
        capa = capa.to('F/g')
        return capa

    def unit_convert_C_g(self, charge, atom_mass =  12.011, n_atom=3620):
        """convert e/amu to C/g

        Args:
            charge (float): the value of elementary charge
            atom_mass (float): the mass for one atom, unit is amu
            n_atom (int): the number of atoms

        Returns:
            float: the value of capacitance (C/g)
        """
        import unyt as unit
        charge_value = charge * unit.qp
        cdc_mass = atom_mass * n_atom * unit.amu
        e_g = charge_value/cdc_mass
        e_g = e_g.to('C/g')
        return e_g