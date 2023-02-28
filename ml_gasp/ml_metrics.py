#!/usr/bin python
"""
Usage: ml_metrics.py [OPTIONS]

  Obtain ML metrics such as the learning curve.

Options:
  --garun_directory DIRECTORY     Path to directory containing GASP run data
                                  [default: /home/salil.bavdekar/ml-
                                  gasp/ml_gasp]
  --regressor [KRR|SVR]           [default: SVR]
  --target [Energy|Formation Energy|Hardness]
                                  [default: Energy]
  --learning_curve                Flag to plot the learning curve
  --help                          Show this message and exit.
  """
import json
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

import constants
import prepare_ml_data
import train_model


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
    "--regressor",
    type=click.Choice(["KRR", "SVR"], case_sensitive=False),
    default="SVR",
    show_default=True,
)
@click.option(
    "--target",
    type=click.Choice(["Energy", "Formation Energy", "Hardness"], case_sensitive=False),
    default="Energy",
    show_default=True,
)
@click.option(
    "--learning_curve",
    is_flag=True,
    help="Flag to plot the learning curve",
)
def main(garun_directory, regressor, target, learning_curve):
    """
    Obtain ML metrics such as the learning curve.
    """
    if target == "Energy":
        target = "Energy per atom"
    ml_metrics(garun_directory, regressor, target, learning_curve)
    print(f"Finished.")


def ml_metrics(garun_directory, regressor, target, learning_curve):
    ml_dir = garun_directory / constants.ML_DIR_NAME
    ml_dir.mkdir(exist_ok=True)

    # Set up logging
    script_name = Path(__file__).stem
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=ml_dir / f"{script_name}.log",
        filemode="w",
        level=logging.INFO,
    )

    # Read pickle with descriptors and targets
    try:
        logging.info("Reading prepared data")
        df = pd.read_pickle(ml_dir / constants.PREPARED_DATA_PKL_NAME)
    except FileNotFoundError:
        logging.warning("Prepared data not found. Running prepare_ml_data.py")
        df = prepare_ml_data.prepare_ml_data(garun_directory)
    else:
        logging.info("Finished obtaining prepared data")

    # Extract descriptors and targets
    X = np.vstack(df["Descriptor"])
    y = df[target].to_numpy()

    # Create model
    logging.info(f"Creating {regressor.upper()} model")
    if regressor.upper() == "SVR":
        ML_model = train_model.create_SVR_model()
    elif regressor.upper() == "KRR":
        ML_model = train_model.create_KRR_model()
    else:
        logging.error("Unsupported ML method!")

    # Learning curve
    if learning_curve:
        logging.info("Plotting learning curve")
        fig_learning_curve, learning_curve_dict = plot_learning_curve(ML_model, X, y)

        learning_curve_png = (
            ml_dir
            / f"learning_curve_{target.replace(' ', '').lower()}_{regressor.upper()}.png"
        )
        fig_learning_curve.savefig(learning_curve_png, dpi=300)

        learning_curve_json = (
            ml_dir
            / f"learning_curve_{target.replace(' ', '').lower()}_{regressor.upper()}.json"
        )
        with open(learning_curve_json, "w") as f:
            json.dump(learning_curve_dict, f, indent=4)

        logging.info(
            f"Finished. Saved to {learning_curve_png} and {learning_curve_json}"
        )

    logging.info("Finished obtaining ML metrics")


def plot_learning_curve(ML_model, X, y, cv=5):
    """
    Plot learning curve for ML model.

    Parameters
    ----------
    ML_model : sklearn.model_selection.RandomizedSearchCV
        ML model.
    X : numpy.ndarray
        Training data.
    y : numpy.ndarray
        Training target.
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Learning curve figure.
    """
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator=ML_model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        n_jobs=-1,
        return_times=True,
    )

    learning_curve_dict = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "fit_times": fit_times,
        "score_times": score_times,
    }

    scores = {
        "train_mean": np.mean(train_scores, axis=1),
        "train_std": np.std(train_scores, axis=1),
        "test_mean": np.mean(test_scores, axis=1),
        "test_std": np.std(test_scores, axis=1),
    }
    times = {
        "train_mean": np.mean(fit_times, axis=1),
        "train_std": np.std(fit_times, axis=1),
        "test_mean": np.mean(score_times, axis=1),
        "test_std": np.std(score_times, axis=1),
    }

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), sharex=True)

    metrics = (scores, times)
    for ax_idx, metric in enumerate(metrics):
        axs[ax_idx].plot(
            train_sizes,
            metric["train_mean"],
            color="blue",
            marker="-s",
            markersize=5,
            label="Training",
        )

        axs[ax_idx].fill_between(
            train_sizes,
            metric["train_mean"] + metric["train_std"],
            metric["train_mean"] - metric["train_std"],
            alpha=0.15,
            color="blue",
        )

        axs[ax_idx].plot(
            train_sizes,
            metric["test_mean"],
            color="green",
            linestyle="-o",
            marker="s",
            markersize=5,
            label="Test",
        )

        axs[ax_idx].fill_between(
            train_sizes,
            metric["test_mean"] + metric["test_std"],
            metric["test_mean"] - metric["test_std"],
            alpha=0.15,
            color="green",
        )

        axs[ax_idx].legend(loc="best")
        axs[ax_idx].grid()

    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("Model score")
    axs[1].set_ylabel("Fit time (s)")
    fig.supxlabel("Number of training samples")

    return fig, learning_curve_dict


if __name__ == "__main__":
    main()
