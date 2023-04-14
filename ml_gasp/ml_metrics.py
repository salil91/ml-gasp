#!/usr/bin python
"""
Usage: ml_metrics.py [OPTIONS]

  Obtain ML metrics such as the learning curve.

Options:
  --garun_directory DIRECTORY     Path to directory containing GASP run data
                                  [default: Current working directory]
  --regressor [KRR|SVR]           [default: SVR]
  --target [Energy|Formation_Energy|Hardness]
                                  [default: Energy]
  --validation_curve [alpha|gamma|epsilon|C|None]
                                  [default: None]
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
from sklearn.preprocessing import StandardScaler

import constants
import prepare_ml_data


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
    "--target",
    type=click.Choice(["Energy", "Formation_Energy", "Hardness"], case_sensitive=False),
    default="Energy",
    show_default=True,
)
@click.option(
    "--regressor",
    type=click.Choice(["KRR", "SVR"], case_sensitive=False),
    default="SVR",
    show_default=True,
)
@click.option(
    "--validation_curve",
    type=click.Choice(["alpha", "gamma", "epsilon", "C", "None"], case_sensitive=False),
    default="None",
    show_default=True,
)
@click.option(
    "--learning_curve",
    is_flag=True,
    help="Flag to plot the learning curve",
)
def main(garun_directory, target, regressor, validation_curve, learning_curve):
    """
    Obtain ML metrics such as the learning curve.
    """
    if target == "Energy":
        target = "Energy per atom"
    if validation_curve == "None":
        validation_curve = None
    print(f"Target: {target}")
    print(f"Regressor: {regressor}")
    print(f"Learning curve: {learning_curve}")
    print(f"Validation curve: {validation_curve}")
    ml_metrics(garun_directory, target, regressor, validation_curve, learning_curve)
    print(f"Finished obtaining ML metrics.")


def ml_metrics(garun_directory, target, regressor, validation_curve, learning_curve):
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
    logging.info(f"Target: {target}")
    logging.info(f"Regressor: {regressor}")
    logging.info(f"Learning curve: {learning_curve}")
    logging.info(f"Validation curve: {validation_curve}")

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

    # Feature scaling
    logging.info("Scaling features")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Validation curve
    if validation_curve:
        logging.info(f"Plotting validation curve for parameter {validation_curve}")
        fig_validation_curve, validation_curve_dict = plot_validation_curve(
            regressor,
            X,
            y,
            parameter=validation_curve,
        )

        validation_curve_fname = f"validation_curve_{target.replace(' ', '').lower()}_{regressor.upper()}_{validation_curve}"
        validation_curve_png = ml_dir / f"{validation_curve_fname}.png"
        fig_validation_curve.savefig(validation_curve_png, dpi=300)

        validation_curve_json = ml_dir / f"{validation_curve_fname}.json"
        with open(validation_curve_json, "w") as f:
            json.dump(validation_curve_dict, f, indent=4, cls=constants.NumpyEncoder)

        logging.info(f"Saved validation curve image to {validation_curve_png}")
        logging.info(f"Saved validation curve data to {validation_curve_json}")

    # Learning curve
    if learning_curve:
        logging.info("Plotting learning curve")
        fig_learning_curve, learning_curve_dict = plot_learning_curve(regressor, X, y)

        learning_curve_fname = (
            f"learning_curve_{target.replace(' ', '').lower()}_{regressor.upper()}"
        )
        learning_curve_png = ml_dir / f"{learning_curve_fname}.png"
        fig_learning_curve.savefig(learning_curve_png, dpi=300)

        learning_curve_json = ml_dir / f"{learning_curve_fname}.json"
        with open(learning_curve_json, "w") as f:
            json.dump(learning_curve_dict, f, indent=4, cls=constants.NumpyEncoder)

        logging.info(f"Saved learning curve image to {learning_curve_png}")
        logging.info(f"Saved learning curve data to {learning_curve_json}")

    logging.info("Finished obtaining ML metrics")


def plot_learning_curve(regressor, X, y, cv=5):
    """
    Plot learning curve for ML model.

    Parameters
    ----------
    regressor : str
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
    from sklearn.model_selection import learning_curve

    import train_model

    # Create model
    logging.info(f"Creating {regressor.upper()} model")
    if regressor.upper() == "SVR":
        ML_model = train_model.create_SVR_model()
    elif regressor.upper() == "KRR":
        ML_model = train_model.create_KRR_model()
    else:
        logging.error("Unsupported ML method!")

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator=ML_model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="r2",
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
            linestyle="-",
            marker="s",
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
            linestyle="-",
            marker="o",
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


def plot_validation_curve(regressor, X, y, parameter, cv=5):
    """
    Plot validation curve for ML model.

    Parameters
    ----------
    regressor : str
        ML model.
    X : numpy.ndarray
        Training data.
    y : numpy.ndarray
        Training target.
    parameter : str
        Parameter to vary.
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Validation curve figure.
    """
    from sklearn.model_selection import validation_curve

    # Create model
    if regressor.upper() == "SVR":
        from sklearn.svm import SVR

        ML_model = SVR(kernel="rbf")
    elif regressor.upper() == "KRR":
        from sklearn.kernel_ridge import KernelRidge

        ML_model = KernelRidge(kernel="rbf")
    else:
        logging.error("Unsupported regressor!")

    param_range = np.logspace(-5, 5, 11)
    train_scores, test_scores = validation_curve(
        estimator=ML_model,
        X=X,
        y=y,
        param_name=parameter,
        param_range=param_range,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    validation_curve_dict = {
        "parameter": parameter,
        "param_range": param_range,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }

    scores = {
        "train_mean": np.mean(train_scores, axis=1),
        "train_std": np.std(train_scores, axis=1),
        "test_mean": np.mean(test_scores, axis=1),
        "test_std": np.std(test_scores, axis=1),
    }

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    ax.plot(
        param_range,
        scores["train_mean"],
        color="blue",
        linestyle="-",
        marker="s",
        markersize=5,
        label="Training",
    )

    ax.fill_between(
        param_range,
        scores["train_mean"] + scores["train_std"],
        scores["train_mean"] - scores["train_std"],
        alpha=0.15,
        color="blue",
    )

    ax.plot(
        param_range,
        scores["test_mean"],
        color="green",
        linestyle="-",
        marker="o",
        markersize=5,
        label="Test",
    )

    ax.fill_between(
        param_range,
        scores["test_mean"] + scores["test_std"],
        scores["test_mean"] - scores["test_std"],
        alpha=0.15,
        color="green",
    )

    ax.legend(loc="best")
    ax.grid()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Model score")
    ax.set_xlabel(parameter)
    ax.set_xscale("log")

    return fig, validation_curve_dict


if __name__ == "__main__":
    main()
