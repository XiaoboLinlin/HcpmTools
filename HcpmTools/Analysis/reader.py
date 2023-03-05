import numpy as np

class read():
    def logfile_reader(self, file):
        ### read log file from lammps
        with open(file,"r") as fi:
            lines = []
            copy = False
            for ln in fi:
                if ln.startswith("Step"):
                    copy = True
                    continue
                if ln.startswith("Loop") or ln.startswith("Lost"):
                    copy = False
                if ln.startswith("if \"${if_restart} == 0\" then \"log log.2\""):
                    copy = False
                if ln.endswith("if \"${if_restart} == 0\" then \"log log.2\""):
                    copy = False
                # if ln.startswith(""):
                #     copy = False
                if ln.startswith("WARN"):
                    continue
                if copy:
                    lines.append(ln)
        data = np.array(lines)  
        df = np.loadtxt(data)
        return df
    def profile_reader(self, file):
        ### read potential file from atc
        with open(file,"r") as fi:
            lines = []
            for ln in fi:
                if ln.startswith(" "):
                    print('yes')
                    lines.append(ln)
        data = np.array(lines)  
        df = np.loadtxt(data)
        return df