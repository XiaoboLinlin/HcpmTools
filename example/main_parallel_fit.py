import math
from timebudget import timebudget
import multiprocessing
import os
import signac
import numpy as np
project = signac.get_project()
from HcpmTools.ParaFit.PdfoFit import PdfoFit
from pdfo import pdfo, Bounds, LinearConstraint, NonlinearConstraint

def make_boundary(x):
    lower_bound = []
    for i in x:
        if i == 1.979:
            lower_bound.append(0.1)
        else:
            lower_bound.append(1)
    
    upper_bound = []
    for i in x:
        if i == 1.979:
            upper_bound.append(50)
        else:
            upper_bound.append(200)
    return lower_bound, upper_bound


x = [1.979, 28.5, 
    1.979, 28.5,
    1.979, 28.5,
    1.979, 28.5,
    1.979, 28.5]



result_file = '/raid6/homes/linx6/project/self_project/mxene_related/pdfo_hcpm/src/fit_result/result3.txt'
process_file = open(result_file, 'w')
print("hello, fit starts... \n", file = process_file)
process_file.close()

# print('\n3. Bound constraints', file=open(result_file, "a"))
print('\n3. Bound constraints', flush=True)
lower_bound, upper_bound = make_boundary(x)
bounds = Bounds(lower_bound, upper_bound)
print(bounds,flush=True)
print('start to calcualte',flush=True)
# cases = [0, 0.1, 0.2, 0.3]
cases = [0, 0.1]
print('case', cases)
pdfofit = PdfoFit(exe_path = '/raid6/homes/linx6/install_software/lammps_27May2021/build/lmp_mpi', lmp_input="/raid6/homes/linx6/project/self_project/mxene_related/pdfo_hcpm/lammps_input/in.data_fit", result_file=result_file, cases = cases)
res = pdfo(pdfofit.parallel_avg_sum_square, x, bounds = bounds)
# print(res, file=open(result_file, "a"), flush=True)
print(res, flush=True)


