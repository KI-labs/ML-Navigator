import logging
import os

import numpy as np
from sklearn import linear_model

from training.model_evaluator import evaluate_model

logger = logging.getLogger(__name__)
formatting = (
    "%(asctime)s: %(levelname)s: File:%(filename)s Function:%(funcName)s Line:%(lineno)d "
    "message:%(message)s"
)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/logs.log"),
    level=logging.INFO,
    format=formatting,
)


def training_for_optimizing(alpha_i: float, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
                            help_text: str) -> float:
    """Trainer

    This function trains the ridge linear regression model given a certain alpha.

    :param float alpha_i: A hyperparameter which used by the ridge linear regression to avoid over-fitting.
    :param np.array x_train: The values of the features which are used to train the model to predict the target y_train
    :param np.array y_train: The target values which are used to train the model.
    :param np.array x_test: The values of the features which are used to evaluate the model by predicting the target
            y_test
    :param np.array y_test: The target values which are used to evaluate the performance of the model based on the
            coefficient of determination R2
    :param str help_text: A string to show useful information about the training cross-validation method
    :return:
            - r2_linear - Coefficient of determination for a given alpha and testing dataset
    """

    linear_regression_model = linear_model.Ridge(alpha=alpha_i)
    model = linear_regression_model.fit(x_train, y_train)

    metrics_summary = evaluate_model(model,
                                     xs=[x_test],
                                     ys=[y_test],
                                     labels=[""],
                                     metrics=["r2_score"])

    r2_linear = metrics_summary['r2_score ']

    return r2_linear


def get_best_alpha_split(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> float:
    """ alpha optimizer for two datasets split

    This function finds the best alpha value based on the coefficient of determination.
    This function will be replaced by a native optimization method from other packages

    :param np.array x_train: The values of the features which are used to train the model to predict the target y_train
    :param np.array y_train: The target values which are used to train the model.
    :param np.array x_test: The values of the features which are used to evaluate the model by predicting the target
            y_test
    :param np.array y_test: The target values which are used to evaluate the performance of the model based on the
            coefficient of determination R2

    :return:
            best_alpha: The alpha value that maximize the coefficient of determination R2
    """

    help_text = "Testing dataset from splitting"

    # generate a list of (alpha, r2_score) pairs
    candidates = [(alpha, training_for_optimizing(alpha, x_train, y_train, x_test, y_test, help_text))
                  for alpha in np.logspace(-3, 3, 19)]

    best_alpha, _ = max(candidates, key=lambda x: x[1])

    logger.info("Alpha optimization for split cross-validation is finished")

    return best_alpha


def get_best_alpha_kfold(kfold, train_array: np.array, target: np.array):
    """ alpha optimizer for K-Fold cross validation

    :param kfold:
    :param train_array: The values of the features which are used to train the model to predict the target target
    :param target: The target values which are used to train the model.

    :return:
            best_alpha: The alpha value that maximize the coefficient of determination R2
    """

    # generate a list of (alpha, r2_score) pairs
    candidates = [
        (alpha,
         np.mean([training_for_optimizing(alpha,
                                          train_array[train],
                                          target[train],
                                          train_array[test],
                                          target[test],
                                          f"dataset kfold {idx}")
                  for idx, (train, test) in enumerate(kfold.split(train_array, target))])
         ) for alpha in np.logspace(-3, 3, 19)
    ]

    best_alpha, _ = max(candidates, key=lambda x: x[1])

    logger.info("Alpha optimization for Kfold cross-validation is finished")

    return best_alpha
