#!/usr/bin python
"""
Usage: prepare_ml_data.py [OPTIONS]

  Create the data array for ML. The data array is saved as a pickle file in
  the ML run directory. Columns: File IDs, structure, molar fraction, energy
  per atom, formation energy, hardness, RDF matrix, and ADF matrix.

  The ML directory is created if it does not exist. The relaxed structures are
  read from the relax directory. If the relax directory does not exist, it is
  created by running get_relaxed_data.py.

Options:
  --garun_directory DIRECTORY  Path to directory containing GASP run data
                               [default: Current working directory]
  --help                       Show this message and exit.
"""

import itertools
import logging
from pathlib import Path

import click
import constants
import get_relaxed_data
import numpy as np
import pandas as pd
import yaml
from pandarallel import pandarallel
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp import Poscar


@click.command()
@click.option(
    "--garun_directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    help="Path to directory containing GASP run data",
    default=".",
    show_default=True,
)
def main(garun_directory):
    """
    Create the data array for ML. The data array is saved as a pickle file in the ML run directory.
    Columns: File IDs, structure, molar fraction, energy per atom, formation energy, hardness, RDF matrix, and ADF matrix.

    The ML directory is created if it does not exist. The relaxed structures are read from the relax directory. If the relax directory does not exist, it is created by running get_relaxed_data.py.
    """
    print("Preparing ML data")
    prepare_ml_data(garun_directory)
    print("Finished preparing ML data")


def prepare_ml_data(garun_directory):
    """
    Returns:
        df (DataFrame): DataFrame with prepared ML data
    """
    relax_dir = garun_directory / constants.RELAX_DIR_NAME
    ml_dir = garun_directory / constants.ML_DIR_NAME
    ml_dir.mkdir(exist_ok=True)

    # Set up logging
    script_name = Path(__file__).stem
    log_path = ml_dir / f"{script_name}.log"
    Path(log_path).unlink(missing_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=log_path,
        filemode="w",
        level=logging.INFO,
    )
    logging.info(f"Run directory: {garun_directory}")

    if not relax_dir.exists():
        logging.warning("Relaxations not found. Preparing ML files")
        get_relaxed_data.get_relaxed_data(garun_directory, hardness=True)

    # Read ga_parameters for element list
    ga_file = garun_directory / "ga_parameters"
    logging.info(f"Reading elements from {ga_file}")
    with open(ga_file, "r") as f:
        ga_parameters = yaml.safe_load(f)
    elements = tuple(ga_parameters["EnergyCode"]["vasp"]["potcars"].keys())
    logging.info(f"Finished. Elements: {elements}")

    # Get file IDs from relaxation directory
    logging.info(f"Getting file IDs from {relax_dir}")
    if not list(relax_dir.glob(f"*.{constants.STRUCTURE_EXT}")):
        logging.warning("Relaxed structures not found. Running get_relaxed_data.py")
        get_relaxed_data.main(garun_directory)
    file_IDs = [f.stem for f in relax_dir.glob(f"*.{constants.STRUCTURE_EXT}")]
    logging.info(f"Number of structures: {len(file_IDs)}")

    # Create DataFrame
    df = pd.DataFrame()
    df["File ID"] = sorted(file_IDs)
    df[["GASP ID", "Run ID"]] = df["File ID"].str.split("_", n=2, expand=True)
    df = df.astype({"GASP ID": int, "Run ID": int})

    # Get Pymatgen Structure objects
    logging.info("Getting structures")
    df["Structure"] = df["File ID"].apply(
        get_structure,
        relax_dir=relax_dir,
    )
    logging.info("Finished")

    logging.info("Calculating molar fractions")
    # Calculate the molar fractions of elem_A and elem_B in the structure
    df["Molar Fraction"] = df["Structure"].apply(calc_mol_frac, elements=elements)
    df = pd.concat(
        [df.drop(columns=["Molar Fraction"]), df["Molar Fraction"].apply(pd.Series)],
        axis=1,
    )
    logging.info("Finished")
    
    logging.info("Getting energies")
    # Get total energy of the structure
    df["Total Energy"] = df["File ID"].apply(
        get_energy,
        axis=1,
        relax_dir=relax_dir,
    )
    logging.info("Finished.")

    logging.info("Calculating average energy per atom")
    # Get average energy per atom for the structures
    df["Energy per atom"] = df.apply(
        calc_epa,
        axis=1,
    )
    logging.info("Finished.")

    # Get the reference energies for elem_A and elem_B
    # i.e. the lowest avg_energy of pure element structures in the dataset
    logging.info("Getting reference energies")
    ref_energies = get_ref_energies(df, elements)
    logging.info(f"Finished. Reference energies: {ref_energies}")

    # Calculate the formation energies from the total energies and reference energies
    logging.info("Calculating formation energies")
    df["Formation Energy"] = df.apply(
        calc_formation_energy, axis=1, elements=elements, ref_energies=ref_energies
    )
    logging.info("Finished")

    # Get the hardness data from the .hardness files
    logging.info("Getting hardness data")
    df["Hardness"] = df["File ID"].apply(
        get_hardness,
        relax_dir=relax_dir,
    )
    logging.info("Finished")

    # Get the global RDF and ADF matrices (one per structure)
    logging.info(
        "Getting RDF and ADF matrices and concatenating them into a single descriptor"
    )
    pandarallel.initialize()
    df["Descriptor"] = df["Structure"].parallel_apply(get_RDFADF, elements=elements)
    logging.info("Finished")

    # Save the DataFrame
    df.set_index("File ID", inplace=True)
    pickle_file = ml_dir / constants.PREPARED_DATA_PKL_NAME
    logging.info(f"Saving prepared DataFrame to {pickle_file}")
    df.to_pickle(pickle_file)
    logging.info("Finished")

    return df


def get_structure(file_ID, relax_dir):
    """
    Get Pymatgen structure object from POSCAR file.

    Args:
        file_ID: File ID of the structure.

        relax_dir: Path to the directory containing the relaxed structures.

    Returns:
        struct: Pymatgen structure object.
    """
    poscar_path = relax_dir / f"{file_ID}.{constants.STRUCTURE_EXT}"
    struct = Poscar.from_file(
        poscar_path, check_for_POTCAR=False, read_velocities=False
    ).structure

    return struct


def calc_mol_frac(structure, elements):
    """
    Calculate the molar fraction of each element in the structure.

    Args:
        structure: Pymatgen structure object.

        elements: list of elements in the dataset.

    Returns:
        molar_frac: dictionary of molar fractions. Keys are "X_elem_A" and "X_elem_B".
    """
    molar_frac = {}
    for element in elements:
        elem_frac = structure.composition.get_atomic_fraction(Element(element))
        molar_frac[f"X_{element}"] = elem_frac

    return molar_frac


def get_energy(file_ID, relax_dir):
    """
    Calculate the average energy per atom for the structure.

    Args:
        file_ID: File ID of the structure.

        relax_dir: Path to the directory containing the relaxed structures.

    Returns:
        epa: average energy per atom.
    """
    with open(relax_dir / f"{file_ID}.{constants.ENERGY_EXT}") as f:
        energy = float(f.read())

    return energy
    

def calc_epa(row):
    """
    Calculate the average energy per atom for the structure.

    Args:
        row: row of the DataFrame.

    Returns:
        epa: average energy per atom.
    """
    structure = row["Structure"]
    epa = row["Total Energy"] / len(structure)

    return epa


def get_ref_energies(df, elements):
    """
    Get the reference energies for each element.

    Args:
        df: DataFrame containing the data.

        elements: list of elements in the dataset.

    Returns:
        ref_energies: dictionary of reference energies for each element. Keys are "elem_A" and "elem_B".
    """
    ref_energies = {}
    for element in elements:
        df_pure_element = df[df[f"X_{element}"] == 1.0]
        min_energy = df_pure_element["Energy per atom"].min()
        ref_energies[element] = min_energy

    return ref_energies


def calc_formation_energy(row, elements, ref_energies):
    """
    Calculate the formation energy for the structure.

    Args:
        row: row of the DataFrame.

        elements: list of elements in the dataset.

        ref_energies: dictionary of reference energies for each element.

    Returns:
        formation_energy: formation energy of the structure.
    """
    ref_contributions = [
        row[f"X_{element}"] * ref_energies[element] for element in elements
    ]
    formation_energy = row["Energy per atom"] - np.sum(ref_contributions)

    return formation_energy


def get_hardness(file_ID, relax_dir):
    """
    Get the hardness for the structure.

    Args:
        file_ID: ID of the structure.

        relax_dir: directory containing the .hardness files.

    Returns:
        hardness: hardness of the structure.
    """
    with open(relax_dir / f"{file_ID}.{constants.HARDNESS_EXT}") as f:
        hardness = float(f.read())

    return hardness


def get_RDFADF(structure, elements):
    """
    Calculates the RDF+ADF descriptor for the structure

    Args:
            structure: input structure.
            elements: list of elements in the dataset.
    """

    def calc_tuples(elements):
        """
        Calculate the tuples for the RDF and ADF matrices.

        Args:
            elements: list of elements in the dataset.

        Returns:
            rdf_tup: list of all element pairs for which the partial RDF is calculated.

            adf_tup: list of all element triplets for which the partial ADF is calculated.
        """
        rdf_tup = [
            list(p) for p in itertools.combinations_with_replacement(elements, 2)
        ]

        adf_tup = [list(p) for p in itertools.product(elements, repeat=3)]
        del adf_tup[3]
        del adf_tup[3]

        return rdf_tup, adf_tup

    def getRDF_Mat(cell, RDF_Tup, cutOffRad=7.51, sigma=0.2, stepSize=0.1):

        """
        Calculates the RDF for the structure.

        Args:
            cell: input structure.

            RDF_Tup: list of all element pairs for which the partial RDF is calculated.

            cutOffRad: max. distance up to which atom-atom intereactions are considered.

            sigma: width of the Gaussian, used for broadening

            stepSize:  bin width, binning transforms the RDF into a discrete representation.
        """
        binRad = np.arange(
            0.1, cutOffRad, stepSize
        )  # Make bins based on stepSize and cutOffRad
        numBins = len(binRad)
        numPairs = len(RDF_Tup)
        vec = np.zeros(
            (numPairs, numBins)
        )  # Create a vector of zeros (dimension: numPairs*numBins)

        # Get all neighboring atoms within cutOffRad for alphaSpec and betaSpec
        # alphaSpec and betaSpec are the two elements from RDF_Tup
        for index, pair in enumerate(RDF_Tup):
            alphaSpec = Element(pair[0])
            betaSpec = Element(pair[1])
            hist = np.zeros(numBins)
            neighbors = cell.get_all_neighbors(cutOffRad)

            sites = cell.sites  # All sites in the structue
            indicesA = [
                j[0] for j in enumerate(sites) if j[1].specie == alphaSpec
            ]  # Get all alphaSpec sites in the structure
            numAlphaSites = len(indicesA)
            indicesB = [
                j[0] for j in enumerate(sites) if j[1].specie == betaSpec
            ]  # Get all betaSpec sites in the structure
            numBetaSites = len(indicesB)

            # If no alphaSpec or betaSpec atoms, RDF vector is zero
            if numAlphaSites == 0 or numBetaSites == 0:
                vec[index] = hist
                continue

            alphaNeighbors = [
                neighbors[i] for i in indicesA
            ]  # Get all neighbors of alphaSpec

            alphaNeighborDistList = []
            for aN in alphaNeighbors:
                tempNeighborList = [
                    neighbor for neighbor in aN if neighbor[0].specie == betaSpec
                ]  # Neighbors of alphaSpec that are betaSpec
                alphaNeighborDist = []
                for j in enumerate(tempNeighborList):
                    alphaNeighborDist.append(j[1][1])
                alphaNeighborDistList.append(
                    alphaNeighborDist
                )  # Add the neighbor distances of all such neighbors to a list

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            for aND in alphaNeighborDistList:
                for dist in aND:
                    inds = dist / stepSize
                    inds = int(inds)
                    lowerInd = inds - 5
                    if lowerInd < 0:
                        while lowerInd < 0:
                            lowerInd = lowerInd + 1
                    upperInd = inds + 5
                    if upperInd >= numBins:
                        while upperInd >= numBins:
                            upperInd = upperInd - 1
                    ind = range(lowerInd, upperInd)
                    evalRad = binRad[ind]
                    exp_Arg = 0.5 * (
                        (np.subtract(evalRad, dist) / (sigma)) ** 2
                    )  # Calculate RDF value for each bin
                    rad2 = np.multiply(
                        evalRad, evalRad
                    )  # Add a 1/r^2 normalization term, check paper for descripton
                    hist[ind] += np.divide(np.exp(-exp_Arg), rad2)

            tempHist = (
                hist / numAlphaSites
            )  # Divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
            vec[index] = tempHist

        matrix = np.row_stack(
            (vec[0], vec[1], vec[2])
        )  # Combine all vectors to get RDFMatrix

        return matrix

    def getADF_Mat(cell, ADF_Tup, cutOffRad=7.51, sigma=0.2, stepSize=0.1, k=5.0):
        """
        Calculates the ADF for every structure.

        Args:
            cell: input structure.

            ADF_Tup: list of all element triplets for which the ADF is calculated.

            cutOffRad: max. distance up to which atom-atom intereactions are considered.

            sigma: width of the Gaussian, used for broadening

            stepSize: bin width, binning transforms the ADF into a discrete representation.

            k: parameter for the logistic cutoff function.
        """
        binRad = np.arange(-1, 1, stepSize)  # Make bins based on stepSize
        numBins = len(binRad)
        numTriplets = len(ADF_Tup)
        vec = np.zeros(
            (numTriplets, numBins)
        )  # Create a vector of zeros (dimension: numTriplets*numBins)

        # Get all neighboring atoms within cutOffRad for alphaSpec, betaSpec, and gammaSpec
        # alphaSpec, betaSpec, and gammSpec are the three elements from ADF_Tup
        for index, triplet in enumerate(ADF_Tup):
            alphaSpec = Element(triplet[0])
            betaSpec = Element(triplet[1])
            gammaSpec = Element(triplet[2])
            hist = np.zeros(numBins)
            neighbors = cell.get_all_neighbors(cutOffRad)

            sites = cell.sites  # All sites in the structue
            indicesA = [
                j[0] for j in enumerate(sites) if j[1].specie == alphaSpec
            ]  # Get all alphaSpec sites in the structure
            numAlphaSites = len(indicesA)
            indicesB = [
                j[0] for j in enumerate(sites) if j[1].specie == betaSpec
            ]  # Get all betaSpec sites in the structure
            numBetaSites = len(indicesB)
            indicesC = [
                j[0] for j in enumerate(sites) if j[1].specie == gammaSpec
            ]  # Get all gammaSpec sites in the structure
            numGammaSites = len(indicesC)

            # If no alphaSpec or betaSpec or gammsSpec atoms, RDF vector is zero
            if numAlphaSites == 0 or numBetaSites == 0 or numGammaSites == 0:
                vec[index] = hist
                continue

            betaNeighbors = [
                neighbors[i] for i in indicesB
            ]  # Neighbors of betaSpec only

            alphaNeighborList = []
            for bN in betaNeighbors:
                tempalphaNeighborList = [
                    neighbor for neighbor in bN if neighbor[0].specie == alphaSpec
                ]  # Neighbors of betaSpec that are alphaSpec
                alphaNeighborList.append(
                    tempalphaNeighborList
                )  # Add all such neighbors to a list

            gammaNeighborList = []
            for bN in betaNeighbors:
                tempgammaNeighborList = [
                    neighbor for neighbor in bN if neighbor[0].specie == gammaSpec
                ]  # Neighbors of betaSpec that are gammaSpec
                gammaNeighborList.append(
                    tempgammaNeighborList
                )  # Add all such neighbors to a list

            # Calculate cosines for every angle ABC using side lengths AB, BC, AC
            cosines = []
            f_AB = []
            f_BC = []
            for B_i, aN in enumerate(alphaNeighborList):
                for i in range(len(aN)):
                    for j in range(len(gammaNeighborList[B_i])):
                        AB = aN[i][1]
                        BC = gammaNeighborList[B_i][j][1]
                        AC = np.linalg.norm(
                            aN[i][0].coords - gammaNeighborList[B_i][j][0].coords
                        )
                        if AC != 0:
                            cos_angle = np.divide(
                                ((BC * BC) + (AB * AB) - (AC * AC)), 2 * BC * AB
                            )
                        else:
                            continue
                        # Use a logistic cutoff that decays sharply, check paper for details [d_k=3, k=2.5]
                        AB_transform = k * (3 - AB)
                        f_AB.append(np.exp(AB_transform) / (np.exp(AB_transform) + 1))
                        BC_transform = k * (3 - BC)
                        f_BC.append(np.exp(BC_transform) / (np.exp(BC_transform) + 1))
                        cosines.append(cos_angle)

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            for r, ang in enumerate(cosines):
                inds = ang / stepSize
                inds = int(inds)
                lowerInd = inds - 2 + 10
                if lowerInd < 0:
                    while lowerInd < 0:
                        lowerInd = lowerInd + 1
                upperInd = inds + 2 + 10
                if upperInd > numBins:
                    while upperInd > numBins:
                        upperInd = upperInd - 1
                ind = range(lowerInd, upperInd)
                evalRad = binRad[ind]
                exp_Arg = 0.5 * (
                    (np.subtract(evalRad, ang) / (sigma)) ** 2
                )  # Calculate ADF value for each bin
                hist[ind] += np.exp(-exp_Arg)
                hist[ind] += np.exp(-exp_Arg) * f_AB[r] * f_BC[r]

            vec[index] = hist

        matrix = np.row_stack(
            (vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
        )  # Combine all vectors to get ADFMatrix

        return matrix

    rdf_tup, adf_tup = calc_tuples(elements)
    rdf_mat = getRDF_Mat(structure, RDF_Tup=rdf_tup)
    adf_mat = getADF_Mat(structure, ADF_Tup=adf_tup)
    descriptor = np.concatenate((rdf_mat, adf_mat), axis=None)

    return descriptor


if __name__ == "__main__":
    main()
