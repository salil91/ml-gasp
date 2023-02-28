#!/usr/bin/ python
"""
Pass the garun_directory as the first argument
"""

import sys
import os
from multiprocessing import Pool, cpu_count
from pymatgen.io.vasp import Xdatcar, Oszicar

def main():
    garun_directory = sys.argv[1]
    
    if not os.path.exists(os.path.join(garun_directory,"relaxations")):
        os.mkdir(os.path.join(garun_directory,"relaxations"))
    
    GASP_ids = [os.path.splitext(f)[1][1:]
                  for f in os.listdir(garun_directory) 
                  if "POSCAR." in f]
    
    structure_ext = ".poscar"
    energy_ext = ".energy"
    n_cores = cpu_count()
    
    zipped_args = [(GASP_id, garun_directory, structure_ext, energy_ext)
                   for GASP_id in GASP_ids]
    Pool(n_cores).map(prep_ml_files, zipped_args)
    

def prep_ml_files(args):
    GASP_id, garun_directory, structure_ext, energy_ext = args
    # try:
    structure_list = Xdatcar(os.path.join(garun_directory,"temp",GASP_id,"XDATCAR")).structures
    energy_list = [step["E0"] for step in Oszicar(os.path.join(garun_directory,"temp",GASP_id,"OSZICAR")).ionic_steps]
    steps = [x for x in range(len(structure_list))]
    for step in steps:
        # filenames have "step+1" to stay consistent with the 1-based indexing that is used in the XDATCAT and OSZICAR files
        structure_filename = os.path.join(garun_directory,"relaxations",GASP_id+"_"+str(step+1)+structure_ext)
        energy_filename = os.path.join(garun_directory,"relaxations",GASP_id+"_"+str(step+1)+energy_ext)
        structure_list[step].to(fmt="poscar", filename=structure_filename)
        with open(energy_filename, "w") as f:
            f.write(str(energy_list[step]))

    # except:
    #     print("noFile    " + GASP_id)


if __name__ == "__main__":
    main()
