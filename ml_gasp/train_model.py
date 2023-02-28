#!/usr/bin python
"""
Usage: train_model.py [OPTIONS]                                                                                     
                                                                                                                    
  Train ML model from pre-split dataset. If the data has not been split,                                            
  split_data.py will be run first.                                                                                  
                                                                                                                    
Options:                                                                                                            
  --garun_directory DIRECTORY     Path to directory containing GASP run data                                        
                                  [default: /home/salil.bavdekar/ml-                                                
                                  gasp/ml_gasp]
  --frac-train FLOAT              Percentage of samples in the training set
                                  [default: 0.8]
  --regressor [KRR|SVR]           [default: SVR]
  --target [Energy|Formation Energy|Hardness]
                                  [default: Energy]
  --help                          Show this message and exit.
"""

import logging
import shelve
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
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
    "--frac-train",
    help="Percentage of samples in the training set",
    default=0.8,
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
def main(garun_directory, frac_train, regressor, target):
    """
    Train ML model.
    """
    if target == "Energy":
        target = "Energy per atom"
    train_model(garun_directory, frac_train, regressor, target)
    print(
        f"Finished training {regressor} model with a {int(frac_train*100)}% training set."
    )


def train_model(garun_directory, frac_train, regressor, target):
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

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, scaler = split_data(df, frac_train, target)

    # Create model
    logging.info(f"Training {regressor.upper()} model")
    if regressor.upper() == "SVR":
        ML_model = create_SVR_model()
    elif regressor.upper() == "KRR":
        ML_model = create_KRR_model()
    else:
        logging.error("Unsupported ML method!")

    # Train ML model and return the best set of hyperparameters for predictions
    ML_model.fit(X_train, y_train)
    ML_best = ML_model.best_estimator_
    logging.info(f"Finished")

    # Plot training predictions
    logging.info("Plotting training predictions")
    y_pred = ML_best.predict(X_train)
    fig_train_pred = plot_predictions(y_train, y_pred)
    train_pred_png = (
        ml_dir
        / ml_dir
        / f"train_pred_{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}.png"
    )
    fig_train_pred.savefig(train_pred_png, dpi=300)
    logging.info(f"Finished. Saved to {train_pred_png}")

    # Test model
    logging.info("Testing model")
    y_pred = ML_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"Finished")
    logging.info(f"R2: {r2}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"MAE: {mae}")

    # Plot testing predictions
    logging.info("Plotting training predictions")
    fig_test_pred = plot_predictions(y_test, y_pred)
    test_pred_png = (
        ml_dir
        / ml_dir
        / f"test_pred_{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}.png"
    )
    fig_test_pred.savefig(test_pred_png, dpi=300)
    logging.info(f"Finished. Saved to {test_pred_png}")

    # Save model
    shelve_model = (
        ml_dir
        / f"{target.replace(' ', '').lower()}_{regressor.upper()}_{int(frac_train*100)}.db"
    )
    logging.info(f"Saving model to {shelve_model}")
    with shelve.open(shelve_model, "c") as db:
        db["model"] = ML_model
        db["target"] = target
        db["frac_train"] = frac_train
        db["regressor"] = regressor
        db["scaler"] = scaler
        db["Predicted"] = y_pred
        db["Expected"] = y_test
        db["R2"] = r2
        db["RMSE"] = rmse
        db["MAE"] = mae
    logging.info(f"Finished")


def split_data(df, frac_train, target):
    """
    Split the prepared data into train and test sets by GASP ID, so that structures from the same relaxation run do not cross between the sets.
    If the data has not been prepared, prepare_ml_data.py will be run first.

    Args:
        df: Dataframe with descriptors and target properties
        frac_train: Fraction of data to use for training (default 0.8)

    Returns:
        df_train: Dataframe with training data
        df_test: Dataframe with testing data
    """
    # Remove nan values in target properties
    logging.info("Removing nan values and positive energies in target properties")
    df.dropna(subset=["Energy per atom", "Formation Energy", "Hardness"], inplace=True)
    positive_energy_idx = df[df["Energy per atom"] >= 0].index
    df.drop(positive_energy_idx, inplace=True)
    logging.info("Finished")

    # Split GASP IDs into training and testing sets
    logging.info("Shuffling and splitting data into training and testing sets")
    gasp_IDs = df["GASP ID"].unique()
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
    logging.info("Finished")

    return X_train, X_test, y_train, y_test, scaler


def create_SVR_model(
    epsilon=0.1,
    cScale=5,
    gScale=0.001,
    n_iter=50,
):
    """
    Create Support Vector Regression model using random search cross-validation.

    Parameters
    ----------
    epsilon : float, default=0.1
        Epsilon parameter for SVR.
    cScale : float, default=5
        Scale parameter for SVR cross-validation.
    gScale : float, default=0.001
        Scale parameter for SVR cross-validation.
    n_iter : int, default=50
        Number of iterations for random search cross-validation.

    Returns
    -------
    ml_model : sklearn.model_selection.RandomizedSearchCV
        ML model.
    """
    from sklearn.svm import SVR

    # TODO: Parameter tuning
    # Use random search CV (5-fold) to select best hyperparameters
    param_dist = {
        "C": scipy.stats.expon(scale=cScale),
        "gamma": scipy.stats.expon(scale=gScale),
        "kernel": ["rbf"],
    }
    ML_model = RandomizedSearchCV(
        estimator=SVR(epsilon=epsilon),
        param_distributions=param_dist,
        cv=4,
        scoring="neg_mean_squared_error",
        n_iter=n_iter,
        n_jobs=-1,
    )

    return ML_model


def create_KRR_model(
    alpha=1,
    gScale=0.001,
    n_iter=50,
):
    """
    Create Kernel Ridge Regression model using random search cross-validation.

    Parameters
    ----------
    alpha : float, default=1
        Scale parameter for KRR cross-validation.
    gScale : float, default=0.001
        Scale parameter for KRR cross-validation.
    n_iter : int, default=50
        Number of iterations for random search cross-validation.

    Returns
    -------
    ml_model : sklearn.model_selection.RandomizedSearchCV
        ML model.
    """
    from sklearn.kernel_ridge import KernelRidge

    # Use random search CV (5-fold) to select best hyperparameters
    param_dist = {
        "alpha": scipy.stats.expon(scale=cScale),
        "gamma": scipy.stats.expon(scale=gScale),
        "kernel": ["rbf"],
    }
    ML_model = RandomizedSearchCV(
        estimator=KernelRidge(),
        param_distributions=param_dist,
        cv=4,
        scoring="neg_mean_squared_error",
        n_iter=n_iter,
        n_jobs=-1,
    )

    return ML_model


def plot_predictions(xx, yy):
    """
    Plot predicted vs. expected values using a gaussian kernel density estimate.

    Parameters
    ----------
    xx : array-like
        Expected values.
    yy : array-like
        Predicted values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    # Calculate the point density
    kde = scipy.stats.gaussian_kde([xx, yy])
    zz = kde([xx, yy])

    # Sort the points by density, so that the densest points are plotted last
    idx = zz.argsort()
    xx, yy, zz = xx[idx], yy[idx], zz[idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    lims = (
        np.amin(np.concatenate((xx, yy), axis=None)),
        np.amax(np.concatenate((xx, yy), axis=None)),
    )
    ax.plot(lims, lims, "k")  # 45 degree line
    ax.scatter(xx, yy, c=zz, s=5)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Expected")
    ax.set_ylabel("Predicted")
    ax.set(xlim=lims, ylim=lims)

    return fig


if __name__ == "__main__":
    main()
