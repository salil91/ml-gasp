#!/usr/bin python
"""
Usage: train_model.py [OPTIONS]

  Train ML model.

Options:
  --garun_directory DIRECTORY     Path to directory containing GASP run data
                                  [default: /home/salil.bavdekar/ml-
                                  gasp/ml_gasp]
  --frac-train FLOAT              Fraction of samples in the training set
                                  [default: 0.8]
  --frac-relax FLOAT              Fraction of unrelaxed structures to sample
                                  [default: 1]
  --target [Energy|Formation_Energy|Hardness]
                                  [default: Formation Energy]
  --regressor [KRR|SVR]           [default: SVR]
  --help                          Show this message and exit.
"""

import json
import logging
import shelve
from pathlib import Path

import click
import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prepare_ml_data
import scipy.stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


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
    default=1,
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
def main(garun_directory, frac_train, frac_relax, target, regressor):
    """
    Train ML model.
    """
    ml_dir = garun_directory / constants.ML_DIR_NAME
    ml_dir.mkdir(exist_ok=True)
    model_fname = (
        f"{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}"
    )
    # Set up logging
    script_name = Path(__file__).stem
    log_path = ml_dir / f"{script_name}_{model_fname}.log"
    Path(log_path).unlink(missing_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=log_path,
        filemode="w",
        level=logging.INFO,
    )

    target = target.replace("_", " ")  # Convert to prepared DataFrame column name
    if target == "Energy":
        target = "Energy per atom"
    print(f"Fraction of training set: {frac_train}")
    print(f"Target: {target}")
    print(f"Regressor: {regressor}")
    df = get_prepared_data(garun_directory, frac_train, target, regressor, ml_dir)
    train_model(df, frac_train, frac_relax, target, regressor, ml_dir, model_fname)
    print(f"Finished training and testing model.")


def get_prepared_data(garun_directory, frac_train, target, regressor, ml_dir):
    """
    Read prepared data from pickle file or run prepare_ml_data.py if pickle file is not found.

    Parameters
    ----------
    garun_directory : Path
        Path to directory containing GASP run data

    frac_train : float
        Fraction of samples in the training set

    target : str
        Target property

    regressor : str
        ML method

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing descriptors and target property
    """
    # Read pickle with descriptors and targets
    try:
        logging.info("Reading prepared data")
        df = pd.read_pickle(ml_dir / constants.PREPARED_DATA_PKL_NAME)
    except FileNotFoundError:
        logging.warning(
            "Prepared data not found. Running prepare_ml_data.py with default parameters."
        )
        df = prepare_ml_data.prepare_ml_data(garun_directory)
    else:
        logging.info("Finished obtaining prepared data")
    logging.info(f"Run directory: {garun_directory}")
    logging.info(f"Fraction of training set: {frac_train}")
    logging.info(f"Target: {target}")
    logging.info(f"Regressor: {regressor}")

    return df


def train_model(df, frac_train, target, regressor, ml_dir, model_fname):
    """
    Train ML model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing descriptors and target property

    frac_train : float
        Fraction of samples in the training set

    target : str
        Target property

    regressor : str
        ML method

    ml_dir : Path
        Path to directory containing ML data

    model_fname : str
        Filename for ML model

    Returns
    -------
    None
    """
    # Split data into training and testing sets
    logging.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test, scaler = split_data(
        df, frac_train=frac_train, target=target
    )

    # Create model
    if regressor.upper() == "SVR":
        ML_model = create_SVR_model()
    elif regressor.upper() == "KRR":
        ML_model = create_KRR_model()
    else:
        logging.error("Unsupported ML method!")

    # Train ML model and return the best set of hyperparameters for predictions
    logging.info(f"Training model")
    ML_model.fit(X_train, y_train)
    ML_best = ML_model.best_estimator_
    ML_params = ML_model.best_params_

    # Save data to new database
    if model_fname is not None:
        model_shelve = ml_dir / f"{model_fname}"
        logging.info(f"Saving split data to {model_shelve}")
        with shelve.open(f"{model_shelve}", "n") as db:
            db["X_train"] = X_train
            db["X_test"] = X_test
            db["y_train"] = y_train
            db["y_test"] = y_test
            db["scaler"] = scaler

        # Plot training predictions
        logging.info("Plotting training predictions")
        y_pred_train = ML_best.predict(X_train)
        fig_train_pred, _ = plot_predictions(y_train, y_pred_train, target)
        train_pred_png = (
            ml_dir
            / ml_dir
            / f"pred_train_{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}.png"
        )
        fig_train_pred.savefig(train_pred_png, dpi=300)
        logging.info(f"Saved to {train_pred_png}")

    # Test model
    logging.info("Testing model")
    y_pred_test = ML_best.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    logging.info(f"R2: {r2}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"MAE: {mae}")

    # Save results
    if model_fname is not None:
        model_json = ml_dir / f"{model_fname}.json"
        model_results_dict = {
            "target": target,
            "regressor": regressor,
            "frac_train": frac_train,
            "num_train": len(y_train),
            "num_test": len(y_test),
            "ML_params": ML_params,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        }
        with open(model_json, "w") as f:
            json.dump(model_results_dict, f, indent=4, cls=constants.NumpyEncoder)
        logging.info(f"Saved results to {model_json}")

        # Plot testing predictions
        logging.info("Plotting testing predictions")
        fig_test_pred, _ = plot_predictions(y_test, y_pred_test, target)
        test_pred_png = (
            ml_dir
            / ml_dir
            / f"pred_test_{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}.png"
        )
        fig_test_pred.savefig(test_pred_png, dpi=300)
        logging.info(f"Saved to {test_pred_png}")

        # Save model
        with shelve.open(f"{model_shelve}", "w") as db:
            db["model"] = ML_model
            db["y_pred_train"] = y_pred_train
            db["y_pred_test"] = y_pred_test
        logging.info(f"Saved model and predictions to {model_shelve}")

    logging.info(f"Finished")

    return r2, rmse, mae


def split_data(df_original, frac_train, frac_relax=1, target="Formation Energy"):
    """
    Split the prepared data into train and test sets by GASP ID, so that structures from the same relaxation run do not cross between the sets.
    If the data has not been prepared, prepare_ml_data.py will be run first.

    Args:
        df_original: Dataframe with all the descriptors and target properties
        frac_train: Fraction of data to use for training (default 0.8)
        frac_relax: Fraction of unrelaxed structures to sample (default 1)

    Returns:
        df_train: Dataframe with training data
        df_test: Dataframe with testing data
    """
    # Remove nan values in target properties
    logging.info("Removing unreasonable and nan values in target properties")
    df_original.dropna(subset=target, inplace=True)
    high_energy_idx = df_original[df_original["Formation Energy"] >= 5].index
    df_original.drop(high_energy_idx, inplace=True)
    if target.lower() == "hardness":
        high_hadness_idx = df_original[df_original["Hardness"] >= 50].index
        df_original.drop(high_hadness_idx, inplace=True)
    logging.info("Finished")

    # Use only part of the unrelaxed structures (usually done in the preparation step)
    if frac_relax < 1:
        df = prepare_ml_data.sample_unrelaxed(df_original, frac_relax)
    else:
        logging.info("Using all unrelaxed structures")
        df = df_original.copy()

    # Split GASP IDs into training and testing sets
    logging.info("Shuffling and splitting data into training and testing sets")
    gasp_IDs = df_original["GASP ID"].unique()
    np.random.shuffle(gasp_IDs)
    num_train_IDs = int(frac_train * len(gasp_IDs))
    num_test_IDs = len(gasp_IDs) - num_train_IDs
    train_IDs = gasp_IDs[:num_train_IDs]
    test_IDs = gasp_IDs[num_train_IDs:]

    logging.info(f"Number of GASP IDS in train set: {num_train_IDs}")
    logging.info(f"Number of GASP IDS in test set: {num_test_IDs}")

    # Create dataframes for training and testing sets
    df_train = df[df["GASP ID"].isin(train_IDs)]
    df_test = df[df["GASP ID"].isin(test_IDs)]
    n_train = len(df_train) / len(df)
    n_test = len(df_test) / len(df)
    logging.info(
        f"Number of structures in train set: {len(df_train)} ({n_train*100:.2f}%)"
    )
    logging.info(
        f"Number of structures in test set: {len(df_test)} ({n_test*100:.2f}%)"
    )

    X_train = np.vstack(df_train["Descriptor"])
    X_test = np.vstack(df_test["Descriptor"])
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()

    # Feature scaling
    logging.info("Scaling features")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info("Finished splitting data")

    return X_train, X_test, y_train, y_test, scaler


def create_SVR_model(
    e_scale=1,
    c_scale=100,
    g_scale=0.01,
    n_iter=500,
):
    """
    Create Support Vector Regression model using random search cross-validation.

    Parameters
    ----------
    e_scale : float, default=1
        Scale for epsilon parameter in SVR cross-validation.
    c_scale : float, default=100
        Scale for C paramter in SVR cross-validation.
    g_scale : float, default=0.01
        Scale for gamma parameter in SVR cross-validation.
    n_iter : int, default=500
        Number of iterations for random search cross-validation.

    Returns
    -------
    ml_model : sklearn.model_selection.RandomizedSearchCV
        ML model.
    """
    from sklearn.svm import SVR

    logging.info(f"Creating SVR model with RandomizedSearchCV")

    # Use random search CV (5-fold) to select best hyperparameters
    param_dist = {
        "epsilon": scipy.stats.expon(scale=e_scale),
        "C": scipy.stats.expon(scale=c_scale),
        "gamma": scipy.stats.expon(scale=g_scale),
        "kernel": ["rbf"],
    }
    ML_model = RandomizedSearchCV(
        estimator=SVR(),
        param_distributions=param_dist,
        cv=5,
        scoring="r2",
        n_iter=n_iter,
        n_jobs=-1,
    )

    return ML_model


def create_KRR_model(
    a_scale=10,
    g_scale=1e4,
    n_iter=1000,
):
    """
    Create Kernel Ridge Regression model using random search cross-validation.

    Parameters
    ----------
    a_scale : float, default=5
        Scale for alpha parameter in KRR cross-validation.
    g_scale : float, default=0.001
        Scale for gamma parameter in KRR cross-validation.
    n_iter : int, default=1000
        Number of iterations for random search cross-validation.

    Returns
    -------
    ml_model : sklearn.model_selection.RandomizedSearchCV
        ML model.
    """
    from sklearn.kernel_ridge import KernelRidge

    logging.info(f"Creating KRR model with RandomizedSearchCV")

    # Use random search CV (5-fold) to select best hyperparameters
    param_dist = {
        "alpha": scipy.stats.expon(scale=a_scale),
        "gamma": scipy.stats.expon(scale=g_scale),
        "kernel": ["rbf"],
    }
    ML_model = RandomizedSearchCV(
        estimator=KernelRidge(),
        param_distributions=param_dist,
        cv=5,
        scoring="r2",
        n_iter=n_iter,
        n_jobs=-1,
    )

    return ML_model


def plot_predictions(xx, yy, target):
    """
    Plot predicted vs. expected values using a gaussian kernel density estimate.

    Parameters
    ----------
    xx : array-like
        Expected values.
    yy : array-like
        Predicted values.
    target : str
        Regression target.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    plt.rcParams.update({"font.size": 16})

    # Calculate the point density
    try:
        kde = scipy.stats.gaussian_kde([xx, yy])
        zz = kde([xx, yy])
    except np.linalg.LinAlgError:
        logging.warning("Could not calculate gaussian KDE")
        zz = np.ones_like(xx)

    # Sort the points by density, so that the densest points are plotted last
    idx = zz.argsort()
    xx, yy, zz = xx[idx], yy[idx], zz[idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    lims = (
        np.amin(np.concatenate((xx, yy), axis=None)),
        np.amax(np.concatenate((xx, yy), axis=None)),
    )
    line_45 = [-100, 100]  # 45 degree line
    ax.plot(line_45, line_45, "k")
    ax.scatter(xx, yy, c=zz, s=5)
    ax.set(xlim=lims, ylim=lims)
    ax.set_aspect("equal", "box")

    if target.lower() == "hardness":
        unit = "GPa"
    else:
        unit = "eV/atom"
    ax.set_xlabel(f"Expected ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")

    return fig, ax


if __name__ == "__main__":
    main()
