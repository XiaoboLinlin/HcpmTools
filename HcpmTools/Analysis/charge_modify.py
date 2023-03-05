import os
import signac
import numpy as np

class Charge_modify:
    """ from the data file and trj file to output new trj with modified atom charges
    """
    def __init__(
        self, 
        pelectrode = None,
        nelectrode = None,
        data_file = None,
        trj_file = None,
        output_file = None
    ):
        """_summary_

        Args:
            pelectrode (int): the molecule number of 
            data_file (str): lammps.data file to provide original atom charge data
            trj_file (str): .lammpstrj file to provide charge info under voltages
            output_file (str): .lammpstrj file with atom charges deducted from the original charge
        """
        self.pelectrode = pelectrode
        self.nelectrode = nelectrode
        self.data_file = data_file
        self.trj_file = trj_file
        self.output_file = output_file
        
    def read_charge(self):
        fin = open(self.data_file, "r")
        fin.close
        linelist = fin.readlines()
        print('nelectrode is ', self.nelectrode)
        print('pelectrode is ', self.pelectrode)
        d_data = dict()
        start = False
        # find the line number for Bonds, which is 2 lines before the last line of 'Atoms'
        for line in linelist:
            if start:
                if len(line.split()) >= 2 and (int(line.split()[1]) == self.pelectrode or int(line.split()[1]) == self.nelectrode): 
                    d_data[line.split()[0]] = line.split()[3]
            try:
                if line.split()[0] == 'Atoms':
                    start = True
                if line.split()[0] == 'Bonds':
                    break
            except:
                continue
        return d_data
        
    def output_trj(self):
        """
        output lammsptrj file with modified atom charges
        
        """
        d_data = self.read_charge()
        fin = open(self.trj_file, "r")
        fout = open(self.output_file, "w")
        fin.close
        linelist = fin.readlines()
        for line in linelist:
            if len(line.split()) >= 7 and line.split()[0] != 'ITEM:':
                if int(line.split()[1]) == self.pelectrode or int(line.split()[1]) == self.nelectrode:
                    new_charge = float(line.split()[-1]) - float(d_data[line.split()[0]])
                    line = line.rsplit(' ', 1)[0] + ' ' + str(round(new_charge,5))+'\n'
            fout.write(line)    
        fout.close()
        print('yes, done')
        