"""
Usage: descriptor_hyperparameters.py [OPTIONS]

  Optimize hyperparameters for descriptor calculation

Options:
  --garun_directory DIRECTORY     Path to directory containing GASP run data
                                  [default: .]
  --frac-train FLOAT              Fraction of samples in the training set
                                  [default: 0.8]
  --frac-relax FLOAT              Fraction of unrelaxed structures to sample
                                  [default: 0.1]
  --target [Energy|Formation_Energy|Hardness]
                                  [default: Formation Energy]
  --regressor [KRR|SVR]           [default: SVR]
  --d-c FLOAT...                  Range for d_c
  --d-k FLOAT...                  Range for d_k
  --k FLOAT...                    Range for k
  --n INTEGER                     Number of iterations  [default: 10]
  --help                          Show this message and exit.
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
    "--frac-train",
    help="Fraction of samples in the training set",
    default=0.8,
    show_default=True,
)
@click.option(
    "--frac-relax",
    help="Fraction of unrelaxed structures to sample",
    default=0.1,
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
def main(garun_directory, frac_train, frac_relax, target, regressor, d_c, d_k, k, n):
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

    # Write hyperparameters and results to csv
    csv_fname = (
        f"{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}"
    )
    csv_path = ml_dir / f"descriptor_hyperparameters_{csv_fname}.csv"
    with open(csv_path, "w") as f:
        fieldnames = ["d_c", "d_k", "k", "r2", "rmse", "mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over random sampling of hyperparameters
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
                garun_directory, frac_relax, d_c=d_c, d_k=d_k, k=k
            )
            r2, rmse, mae = train_model.train_model(
                df,
                frac_train=frac_train,
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
