import logging
import os

import numpy as np
from sklearn import linear_model

from training.regression_model_evaluator import regression_evaluate_model

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


def int_2_float_alpha(alpha: int) -> float:
    """Alpha search step definer

    This function defines the searching step when optimizing alpha using a linear search.

    :param alpha: A hyperparameter which used by the ridge linear regression to avoid over-fitting
    :return:
            The value of alpha after dividing it to a certain number e.g. 10 to define the searching step.
    """

    return alpha / 10.0


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
    _, metrics_summary = regression_evaluate_model(model, x_test, y_test, help_text,
                                                   help_print=False)

    r2_linear = metrics_summary['r2_score']

    return r2_linear


def choosing_alpha(r2_linear: float, r2_linear_max: float, alpha_i: float, best_alpha: float):
    """ Alpha evaluator

    This function check if the current value of alpha gives higher value of coefficient of determination than the
    current value. It updates the values of the best found alpha and r2 if needed.

    :param float r2_linear:
    :param float r2_linear_max: The current maximum coefficient of determination assiosated with the current
                            best_alpha value and testing dataset
    :param float alpha_i: The current value of the hyperparameter which used by the ridge linear regression to avoid
                         over-fitting.
    :param float best_alpha:

    :return:
            - r2_linear_max - The updated maximum coefficient of determination for a given best_alpha and testing dataset
            - best_alpha - The updated alpha value that maximize the r2_linear_max for a given testing dataset
    """

    if r2_linear > r2_linear_max:
        r2_linear_max = r2_linear
        best_alpha = alpha_i

        logger.info(f"New alpha was chosen: {best_alpha}")
        logger.info(f"Maximum r2 for alpha value {best_alpha} is : {r2_linear_max}")

    return r2_linear_max, best_alpha


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

    # Initiate variables
    r2_linear_max = 0
    best_alpha = 0.1

    # simple linear search
    for alpha_i in range(1, 1000):
        # Since the range gives integers only, I got all possible
        # alpha values between 0.1 and 100 as float alpha
        float_alpha = int_2_float_alpha(alpha_i)

        help_text = "Testing dataset from splitting"
        r2_linear = training_for_optimizing(float_alpha, x_train, y_train, x_test, y_test, help_text)

        # Choose alpha based on maximum value of r2
        r2_linear_max, best_alpha = choosing_alpha(r2_linear, r2_linear_max, float_alpha, best_alpha)

    logger.info("Alpha optimization for split cross-validation is finished")
    return best_alpha


def get_best_alpha_kfold(kfold, train_array: np.array, target: np.array, n_fold: int):
    """ alpha optimizer for K-Fold cross validation

    :param kfold:
    :param train_array: The values of the features which are used to train the model to predict the target target
    :param target: The target values which are used to train the model.
    :param n_fold: An integer that refers to the number of folds which will be used for cross-validation.

    :return:
            best_alpha: The alpha value that maximize the coefficient of determination R2
    """

    # initiate the variables
    r2_linear_max = 0
    best_alpha = 0.1

    # simple linear search
    for alpha_i in range(1, 1000):
        # Since the range gives integers only, I got all possible
        # alpha values between 0.1 and 100 as float alpha
        float_alpha = int_2_float_alpha(alpha_i)

        fold_nr = 0  # counter for identifying models
        r2_linear = 0

        for train, test in kfold.split(train_array, target):
            fold_nr += 1

            help_text = f"dataset kfold {fold_nr}"
            r2_linear_i = training_for_optimizing(float_alpha, train_array[train],
                                                  target[train],
                                                  train_array[test],
                                                  target[test],
                                                  help_text)

            r2_linear += r2_linear_i

        r2_linear = round(100 * r2_linear / n_fold, 2)

        # Choose alpha based on maximum value of r2
        r2_linear_max, best_alpha = choosing_alpha(r2_linear, r2_linear_max, float_alpha, best_alpha)

    logger.info("Alpha optimization for Kfold cross-validation is finished")
    return best_alpha
