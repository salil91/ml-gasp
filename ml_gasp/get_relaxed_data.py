#!/usr/bin/ python
"""
Usage: get_relaxed_data.py [OPTIONS]

  Parse GA run directory and separate relaxation structures into .poscar,
  .energy, and .hardness files.

Options:
  --garun_directory DIRECTORY  Path to directory containing GASP run data
                               [default: Current working directory]
  --hardness / --no-hardness   Flag to calculate hardness.  [default:
                               hardness]
  --help                       Show this message and exit.
"""

import logging
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
from intrinsic_hardness import bond_detectors, hardness_calculators
from pymatgen.io.vasp import Oszicar, Xdatcar

import constants


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
@click.option(
    "--hardness/--no-hardness",
    help="Flag to calculate hardness.",
    default=True,
    show_default=True,
)
def main(garun_directory, hardness):
    """
    Parse GA run directory and separate relaxation structures into .poscar, .energy, and .hardness files.
    """
    get_relaxed_data(garun_directory, hardness)
    print("Finished getting relaxed data.")


def get_relaxed_data(garun_directory, hardness):
    # TODO: Intergrate with prepare_ml_data.py and get rid of the need to write all these files.
    relax_dir = garun_directory / constants.RELAX_DIR_NAME
    ml_dir = garun_directory / constants.ML_DIR_NAME
    ml_dir.mkdir(exist_ok=True)
    relax_dir.mkdir(exist_ok=True)

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

    # Get GASP ID from final structures
    logging.info(f"Reading final structures from {garun_directory}")
    gasp_ids = [f.suffix[1:] for f in garun_directory.glob("POSCAR.*")]
    logging.info(f"Found {len(gasp_ids)} GASP IDs")

    # Get structures and energies from relaxation runs
    n_cores = cpu_count()
    zipped_args = [
        (
            gasp_id,
            garun_directory,
            relax_dir,
            hardness,
        )
        for gasp_id in gasp_ids
    ]
    logging.info(f"Getting relaxation runs using {n_cores} cores")
    Pool(n_cores).map(get_relaxation_runs, zipped_args)
    poscar_files = relax_dir.glob(f"*.{constants.STRUCTURE_EXT}")
    logging.info(f"Found {len(list(poscar_files))} relaxed structures")


def get_relaxation_runs(args):
    """
    Parse folders for each GASP structure and return the structures and energies for each relaxation step.

    Args:
        args (tuple): Tuple of arguments
            gasp_id (str): GASP ID
            garun_directory (Path): Path to GA run directory
            relax_dir (Path): Path to directory to store relaxed structures
            hardness (bool): Flag to calculate hardness

    Returns:
        None
    """
    gasp_id, garun_directory, relax_dir, hardness = args

    structure_list = Xdatcar(garun_directory / "temp" / gasp_id / "XDATCAR").structures
    energy_list = [
        step["E0"]
        for step in Oszicar(garun_directory / "temp" / gasp_id / "OSZICAR").ionic_steps
    ]

    for step, (structure, energy) in enumerate(
        zip(structure_list, energy_list), start=1
    ):
        # start=1 to stay consistent with the 1-based indexing that is used in the XDATCAT and OSZICAR files
        file_stem = f"{gasp_id}_{step}"
        structure_filename = relax_dir / f"{file_stem}.{constants.STRUCTURE_EXT}"
        energy_filename = relax_dir / f"{file_stem}.{constants.ENERGY_EXT}"
        structure.to(fmt="poscar", filename=structure_filename)
        with open(energy_filename, "w") as f:
            f.write(f"{energy}")

        if hardness:
            # Calculate and write hardness
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=UserWarning)
                bonds = bond_detectors.detect_bonds(structure)
            intrinsic_hardness = hardness_calculators.calculate_hardness(
                structure, bonds, model="CAS"
            )
            hardness_filename = relax_dir / f"{file_stem}.{constants.HARDNESS_EXT}"
            with open(hardness_filename, "w") as f:
                f.write(f"{intrinsic_hardness}")


if __name__ == "__main__":
    main()
