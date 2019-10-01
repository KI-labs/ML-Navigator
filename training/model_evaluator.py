import logging
import os
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

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

metrics_map = {
    "mean_squared_error": mean_squared_error,
    "r2_score": lambda x1, x2: round(100 * r2_score(x1, x2), 2),
    "accuracy_score": lambda x1, x2: round(accuracy_score(x1, np.around(x2)), 4),
    "roc_auc_score": roc_auc_score
}


def load_all_models(save_models_dir: str, model_type: str, model_i: int):
    """ Model loader

    Load saved models from a given type.

    :param str save_models_dir: directory where the model is saved
    :param str model_type: `regression` or `classification`
    :param int model_i: index used to distinguish models of the same type trained on different datasets.
    :return:
    """

    try:
        with open(os.path.join(save_models_dir, f"{model_type}_{model_i}.pkl"), "rb") as save_model:
            model = pickle.load(save_model)
    except Exception as e:
        logger.error(f"Error is: {e}")
        return None

    return model


def evaluate_model(model,
                   xs: list, ys: list, labels: list,
                   metrics: list):
    """ Model evaluator

    This function shows the value of the matrices R2 and MSE for different datasets when evaluating the trained model.

    :param model: An object created by the training package e.g. Scikit Learn.
    :param list xs: Every element is a np.array of the features that are used to predict the target variable.
    :param list ys: Every element is a np.array of the target variable.
    :param list labels: Every element is a string that is used to label every (x,y) pair and refers to their origin.
    :param list metrics: list of metrics used to evaluate model.

    :return:
            metrics_summary (dict): all metrics from `metrics` applied to all (y, y_pred=model(x)) paris.
    """

    metrics_summary = {}

    for x, y, label in zip(xs, ys, labels):

        y_pred = model.predict(x)

        for metric in metrics:
            try:
                value = metrics_map[metric](y, y_pred)
            except Exception as e:
                logger.error(f"Error during metric calculation: {e}")
                print(f"Error during metric calculation: {e}\n"
                      "metric value set to np.nan ")
                value = np.nan

            metrics_summary[f"{metric} {label}"] = value

    logger.info("Evaluate Model process is finished")

    return metrics_summary
