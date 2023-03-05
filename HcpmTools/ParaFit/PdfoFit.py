import os
import signac
import numpy as np
import MDAnalysis as mda
import mdtraj as md
import parmed as pmd    
import subprocess
import multiprocessing

project = signac.get_project()

class PdfoFit:
    def __init__(
        self,
        exe_path = '/raid6/homes/linx6/install_software/lammps_27May2021/build/lmp_mpi',
        lmp_input = '/raid6/homes/linx6/project/self_project/mxene_related/cpm_dft_mxene/lammps_input',
        result_file = None, 
        cases = None,
        print = True,
        lmp_compile_version = 'gcc'
    ):
        """_summary_

        Args:
            lmp_input (str, optional): _description_. Defaults to '/raid6/homes/linx6/project/self_project/mxene_related/cpm_dft_mxene/lammps_input'.
            cases (_type_, optional): _description_. Defaults to None. if provided, it can calculate the sum_square per case. 
            lmp_compile_version (str): Default to 'gcc', how lammps is compiled, using gcc or intel.
        """
        self.exe_path = exe_path
        self.lmp_input = lmp_input
        self.result_file = result_file
        self.cases = cases
        self.print = print
        self.lmp_compile_version = lmp_compile_version
        
        
    def produce_lmpcpm(self):
            #mpirun -np 1 
        if self.lmp_compile_version == 'gcc':
            lmp_scipt = 'mpirun -np 3 {exe_path} -in {lmp_input} '\
                        '-var mx_tcharge {mx_tcharge} -var kappa {kappa} '\
                        '-var width_H {width_H} -var Aii_H {Aii_H} '\
                        '-var width_O {width_O} -var Aii_O {Aii_O} '\
                        '-var width_Tio {width_Tio} -var Aii_Tio {Aii_Tio} '\
                        '-var width_C {width_C} -var Aii_C {Aii_C} '\
                        '-var width_Tii {width_Tii} -var Aii_Tii {Aii_Tii}'
                        
        elif self.lmp_compile_version == 'intel':
            lmp_scipt = 'mpirun -np 2 {exe_path} -in {lmp_input} '\
                        '-var mx_tcharge {mx_tcharge} -var kappa {kappa} '\
                        '-var width_H {width_H} -var Aii_H {Aii_H} '\
                        '-var width_O {width_O} -var Aii_O {Aii_O} '\
                        '-var width_Tio {width_Tio} -var Aii_Tio {Aii_Tio} '\
                        '-var width_C {width_C} -var Aii_C {Aii_C} '\
                        '-var width_Tii {width_Tii} -var Aii_Tii {Aii_Tii} '\
                        '-sf intel -pk intel 0 omp 2'
                    
        subprocess.run(lmp_scipt.format(
                                exe_path = self.exe_path,
                                lmp_input = self.lmp_input,
                                mx_tcharge = self.mx_tcharge, kappa=self.kappa,
                                width_H = self.width_H, Aii_H = self.Aii_H,
                                width_O = self.width_O, Aii_O = self.Aii_O,
                                width_Tio = self.width_Tio, Aii_Tio =self.Aii_Tio,
                                width_C = self.width_C, Aii_C = self.Aii_C,
                                width_Tii = self.width_Tii, Aii_Tii = self.Aii_Tii
                            ), 
                            shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True
                        )
        
    def sum_square(self, x, job):
        """calculate the sum of square for a single signc job
        
        Args:
            x (_type_): _description_
            job (signac job): _description_

        Returns:
            _type_: _description_
        """
        self.job_workspace = job.workspace()
        self.x = x
        self.width_H, self.Aii_H =x[0], x[1]
        self.width_O, self.Aii_O =x[2], x[3]
        self.width_Tio, self.Aii_Tio= x[4], x[5]
        self.width_C, self.Aii_C=x[6], x[7]
        self.width_Tii, self.Aii_Tii=x[8], x[9]
        self.kappa=1.75
        
        
        
        gro_file = os.path.join(self.job_workspace, 'system.gro')
        structure_gro = pmd.load_file(gro_file)
        u_gro = mda.Universe(structure_gro)
        
        # to get idx for all li atoms
        u_li = u_gro.select_atoms('name Li')
        li_idx = u_li.indices
        
        top_file = os.path.join(self.job_workspace, 'system.top')
        structure_top = pmd.load_file(top_file)
        u_top = mda.Universe(structure_top)
        dft_icharge = u_top.atoms.charges
        li_tcharge = np.sum(dft_icharge[li_idx])
        self.mx_tcharge = -li_tcharge 
        if self.print:
            print('x is {}'.format(self.x), file=open(self.result_file, "a"), flush=True)
        # print('mx_tcharge is {}'.format(mx_tcharge))
        with job:
            self.produce_lmpcpm()
            
        lmp_trj = os.path.join(self.job_workspace, 'customize_conp.lammpstrj')
        lmp_output = np.loadtxt(lmp_trj, skiprows=9)
        lmp_icharge = lmp_output[:, -1]
        
        lmp_mx_icharge = np.delete(lmp_icharge, li_idx, axis = 0)
        dft_mx_icharge = np.delete(dft_icharge, li_idx, axis = 0)
        ## want to minimize the total sum of squares
        total_sum_sdiff = np.sum((lmp_mx_icharge - dft_mx_icharge)**2)
        if self.print:
            print('total_sum_sdiff is {}'.format(total_sum_sdiff), file=open(self.result_file, "a"),flush=True)
        return total_sum_sdiff      
        
    
    def single_sum_sqaure(self, case):
        for job in project.find_jobs({"case": round(case,2)}):
            
            ## make result file in each job
            # result_file = os.path.join(job.workspace(), 'result.txt')
            # process_file = open(result_file, 'w')
            # print("hello, fit starts... \n", file = process_file)
            # process_file.close()
            
            sum_sdiff = self.sum_square(self.x, job=job)
            return sum_sdiff 
            
    def parallel_avg_sum_square(self, x):
        """parallel computing for many cases to get the averaged sum per case

        Args:
            x (_type_): _description_
        """
        
        self.x = x
        pool = multiprocessing.Pool(processes=len(self.cases))
        inputs = self.cases
        outputs = pool.map(self.single_sum_sqaure, inputs)
        # outputs = pool.map(single_sum_sqaure, self.cases)
        avg_sum = np.sum(outputs)/len(self.cases)
        # print("x is {}".format(x))
        print("total_avg: {} ; x is {}".format(avg_sum, self.x), file=open(self.result_file, "a"),flush=True)
        # print("total_avg: {} ; x is {}".format(avg_sum, self.x), flush=True)
        pool.close()
        return avg_sum
        