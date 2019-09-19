import logging
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from IPython.display import display

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


def classification_evaluate_model(model, x_values: np.array, y_values: np.array, key_str: str, help_print: bool = True,
                                  required_metrics: list = ["accuracy_score", "roc_auc_score"]):
    """ Model evaluator

    This function shows the value of the matrices R2 and MSE for different datasets when evaluating the trained model.

    :param list required_metrics: The list of the classification metrics that the user wants to check
    :param model: An object created by the training package e.g. Scikit Learn.
    :param np.array x_values: The values of the features which are used to predict the y_values .
    :param np.array y_values: The target that should be predicted.
    :param str key_str: Help string that tells the user which dataset is used for the evaluation of the model
            performance
    :param bool help_print: If True, information about the model's performance is printed to the console.

    :return:
            | y_prediction: The predicted target using the trained model
            | accuracy: Accuracy classification score.
            | roc_auc:Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    """

    y_prediction = model.predict(x_values)

    accuracy = round(accuracy_score(y_values, np.around(y_prediction)), 4)

    metrics_summary = {
        "accuracy_score": accuracy
    }

    if len(set(y_values)) < 3:
        roc_auc = round(100 * roc_auc_score(y_values, y_prediction), 1)
        metrics_summary["roc_auc_score"] = roc_auc

        if help_print:
            print(f"The quality of the model using the {key_str}")
            if "accuracy_score" in required_metrics:
                print(f"{key_str}: accuracy score: {accuracy}")
            if "roc_auc_score" in required_metrics:
                print(f'{key_str}:  ROC AUC score: {roc_auc} %')

        logger.info("Evaluate Model process is finished")
        return y_prediction, metrics_summary
    else:
        if help_print:
            if "accuracy_score" in required_metrics:
                print(f"{key_str}: accuracy score: {accuracy}")
            if "roc_auc_score" in required_metrics:
                print("ROC AUC score is not available for multi-class classification")

        return y_prediction, metrics_summary


def classification_model_evaluation(data: dict, models_nr: list, save_models_dir: str, model_type: str,
                                    required_metrics: list = ["accuracy_score", "roc_auc_score"]):
    """ Models set evaluator

    The function shows the value of the matrices R2 and MSE for different datasets when evaluating a set of the trained
    models by applying the function `evaluate_model`

    :param required_metrics: The list of the classification metrics that the user wants to check
    :param dict data: A dictionary that contains pandas dataframes as datasets.
    :param list models_nr: A list of indexes that will be used to point to the trained models which will be saved
            locally after training.
    :param str save_models_dir: The path where the models will be saved.
    :param str model_type: The type of model that will be used to fit the data

    :return:
    """

    model = None
    metrics_summary_all = {}

    for data_key, dataframe in data.items():
        try:
            array = data[data_key]["features"].to_numpy()
        except Exception as e:
            print(f"This is xgboost exception. Error: {e}")
            array = data[data_key]["features"]
        target = np.array(data[data_key]["target"])

        accuracy = 0
        roc_auc = 0

        for model_i in models_nr:
            try:
                with open(os.path.join(save_models_dir, f"{model_type}_{model_i}.pkl"), "rb") as save_model:
                    model = pickle.load(save_model)
            except Exception as e:
                logger.error(f"Error is: {e}")

            _, metrics_summary = classification_evaluate_model(model, array, target,
                                                               f"Evaluating the dataset: {data_key}",
                                                               required_metrics=required_metrics)

            metrics_summary_all[f"model_{model_i}"] = dict((f"{k} ({data_key})", v) for k, v in metrics_summary.items()
                                                           if k in required_metrics)

            accuracy += metrics_summary["accuracy_score"]
            roc_auc += metrics_summary.get("roc_auc_score", 0)

            logger.info(f"Model number {model_i} was loaded successfully")

        if "accuracy_score" in required_metrics:
            print(f"{data_key}: The accuracy score: {accuracy / len(models_nr)}")
        if len(set(target)) < 3 and "roc_auc_score" in required_metrics:
            print(f'{data_key}:  ROC AUC score: {roc_auc / len(models_nr)} %')

    metrics_summary_all = pd.DataFrame(metrics_summary_all)
    metrics_summary_all['mean'] = metrics_summary_all.mean(axis=1)
    display(metrics_summary_all)

    logger.info("The evaluation process is completed")
