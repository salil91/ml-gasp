"""
Usage: descriptor_hyperparameters.py [OPTIONS]

  Optimize hyperparameters for descriptor calculation

Options:
  --garun_directory DIRECTORY  Path to directory containing GASP run data
                               [default: /home/salil.bavdekar/ml-gasp/ml_gasp]
  --d-c FLOAT...               Range for d_c
  --d-k FLOAT...               Range for d_k
  --k FLOAT...                 Range for k
  --n INTEGER                  Number of iterations  [default: 10]
  --help                       Show this message and exit.
"""
import click
import random
import logging
from pathlib import Path
import constants
import prepare_ml_data
import train_model
import csv


@click.command()
@click.option(
    "--garun_directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
    help="Path to directory containing GASP run data",
    default=Path.cwd(),
    show_default=True,
)
@click.option(
    "--d-c",
    help="Range for d_c",
    nargs=2,
    type=float,
)
@click.option(
    "--d-k",
    help="Range for d_k",
    nargs=2,
    type=float,
)
@click.option(
    "--k",
    help="Range for k",
    nargs=2,
    type=float,
)
@click.option(
    "--n",
    help="Number of iterations",
    default=10,
    show_default=True,
)
@click.option(
    "--target",
    type=click.Choice(["Energy", "Formation_Energy", "Hardness"], case_sensitive=False),
    default="Formation Energy",
    show_default=True,
)
@click.option(
    "--regressor",
    type=click.Choice(["KRR", "SVR"], case_sensitive=False),
    default="SVR",
    show_default=True,
)
def main(garun_directory, d_c, d_k, k, n, target, regressor):
    """Optimize hyperparameters for descriptor calculation"""
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

    csv_path = ml_dir / "hyperparameters.csv"
    with open(csv_path, "w") as f:
        fieldnames = ["d_c", "d_k", "k", "r2", "rmse", "mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n):
            logging.info(f"Iteration {i+1}/{n}")
            if d_c is not None:
                d_c = round(random.uniform(d_c[0], d_c[1]), 2)
            else:
                d_c = 6.01

            if d_k is not None:
                d_k = round(random.uniform(d_c[0], d_c[1]), 2)
            else:
                d_k = 6.01

            if k is not None:
                k = round(random.uniform(d_c[0], d_c[1]), 2)
            else:
                k = 2.5

            logging.info(f"d_c: {d_c}")
            logging.info(f"d_k: {d_k}")
            logging.info(f"k: {k}")

            df = prepare_ml_data.prepare_ml_data(
                garun_directory, frac_relax=0.1, d_c=d_c, d_k=d_k, k=k
            )
            r2, rmse, mae = train_model.train_model(
                df,
                frac_train=0.8,
                target=target,
                regressor=regressor,
                ml_dir=ml_dir,
                model_fname=None,
            )
            writer.writerow(
                {"d_c": d_c, "d_k": d_k, "k": k, "r2": r2, "rmse": rmse, "mae": mae}
            )


if __name__ == "__main__":
    main()
