#!/usr/bin/ python
"""
This module contains constants used in the ml_gasp package.
"""
import json

# Sub-directories directly under the GA run directory
RELAX_DIR_NAME = "relaxations"
ML_DIR_NAME = "ml_run_data"
PREPARED_DATA_PKL_NAME = "prepared_data.pkl"

# File extensions
STRUCTURE_EXT = "poscar"
ENERGY_EXT = "energy"
HARDNESS_EXT = "hardness"

# Encoder to save NumPy objects in JSONs
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
