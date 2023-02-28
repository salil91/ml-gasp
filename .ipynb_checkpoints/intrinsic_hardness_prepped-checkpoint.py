#!/usr/bin/ python
# coding: utf-8

"""
Intrinsic Harness from GASP results:

This script reads the structures from the .poscar files in the ml_prepped
directory and calculates the intrinsic hardness of the structures based on
the Cheenady model.

Usage: python intrinsic_hardness_prepped.py /path/to/all/relaxations
xxx.poscar files must be in this directory
"""

# Import libraries and read files
import os
import sys

import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import CrystalNN


def main():
    relaxations_directory = sys.argv[1]
    
    GASP_ids = [os.path.splitext(f)[0]
                for f in os.listdir(relaxations_directory) 
                if ".poscar" in f]
    
    hardness_ext = ".hardness"
    
    for GASP_id in GASP_ids:
        hardness = calc_intrinsic_hardness(GASP_id, relaxations_directory)        
        hardness_filename = os.path.join(relaxations_directory,GASP_id+hardness_ext)
        with open(hardness_filename, "w") as f:
            f.write(str(hardness))
        

def calc_intrinsic_hardness(GASP_id, relaxations_directory):
    structure_file = os.path.join(relaxations_directory,GASP_id+".poscar")
    structure = Poscar.from_file(structure_file).structure
    vol = structure.volume

    # Find bonds and calculate bond length
    bonds = []
    for site_index, atom in enumerate(structure):
        nn_object = CrystalNN()
        try:
            neighbors = nn_object.get_nn_info(structure, site_index)
        except:
            continue
        # if not neighbors: continue
        CN1 = nn_object.get_cn(structure, site_index)
        if CN1==0: continue
        EN1 = atom.specie.X
        for neighbor in neighbors:
            if neighbor['site_index'] < site_index: continue
            CN2 = nn_object.get_cn(structure, neighbor['site_index'])
            if CN2==0: continue
            EN2 = neighbor['site'].specie.X
            #bl = math.dist(structure[site_index].coords, neighbor['site'].coords)
            bl = np.linalg.norm(structure[site_index].coords - neighbor['site'].coords)
            bonds.append({"site_1": site_index, "atom_1": atom.specie,
                      "Z_1": atom.specie.Z, "CN_1": CN1, "EN_1": EN1,
                      "site_2": neighbor['site_index'], "atom_2": neighbor['site'].specie,
                      "Z_2": neighbor['site'].specie.Z, "CN_2": CN2, "EN_2": EN2,
                      "bond_length": bl})
    N = len(bonds)

    #Temporary fix for when no bonds are found
    if N == 0:
        H_C = float("nan")
    else:
        # Calculate Intrinsic hardness
        prod_C = 1
        for bond in bonds:
            # Cheenady model
            fi_C = ((bond["EN_1"] - bond["EN_2"]) / (bond["EN_1"] + bond["EN_2"]))**2
            Z_ij = (bond["EN_1"]/bond["CN_1"]) * (bond["EN_2"]/bond["CN_2"])
            prod_C = prod_C * (Z_ij**0.006) * (bond["bond_length"]**-3.18) * np.exp(-2.44*fi_C)

        H_C = (986*(N/vol)**0.844) * prod_C**(1/N)
        
    return H_C


if __name__ == "__main__":
    main()
    
