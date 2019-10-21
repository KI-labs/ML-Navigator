import logging
import os
import pickle

import numpy as np

from training.xgboost_train import xgboost_data_preparation_to_predict

logger = logging.getLogger(__name__)
formatting = (
    "%(asctime)s: %(levelname)s: File:%(filename)s Function:%(funcName)s Line:%(lineno)d "
    "message:%(message)s"
)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/logs.log"),
    level=logging.DEBUG,
    format=formatting,
)


def predict_unseen_data(predict: dict, models_nr: list, save_models_dir: str, model_type: str) -> np.array:
    """ Target prediction

    This function predicts unseen data given in the dictionary `predict`

    :param dict predict: A dictionary that contains pandas dataframes which are unseen data that will be used to
            predict the target
    :param list models_nr: A list of indexes that will be used to point to the trained models which were saved
            locally after training.
    :param str save_models_dir: The path where the models will be saved.
    :param str model_type: The type of the model that was used to fit the data e.g. linear or lightgbm

    :return:
            - y_prediction - A numpy array that contains the predicted target of the unseen data.
    """

    y_prediction = 0
    model = None

    for data_key, dataframe in predict.items():
        if model_type == "xgboost":
            dataframe_i = predict[data_key]["features"]
            dataframe_i.columns = [x for x in range(len(list(dataframe_i.columns)))]
            array = xgboost_data_preparation_to_predict(dataframe_i)
        else:
            array = predict[data_key]["features"].values

        # reset the variable
        y_prediction = 0

        for model_i in models_nr:
            try:
                with open(os.path.join(save_models_dir, f"{model_type}_{model_i}.pkl"), "rb") as save_model:
                    model = pickle.load(save_model)
                logger.info("The model is loaded successfully: " + f"{model_type}_{model_i}.pkl")
            except Exception as e:
                logger.error(f"Loading the model failed. Error is {e}")

            y_prediction += model.predict(array)

        # Stacking the results of different models using the mean value
        y_prediction = y_prediction / len(models_nr)

    return y_prediction


def model_prediction(predict: dict, models_nr: list, save_models_dir: str, model_type: str) -> np.array:
    """Predictor

    This function applies the function `predict_unseen_data` to predict the target using the data in the `predict`.

    :param dict predict: A dictionary that contains pandas dataframes which are unseen data that will be used to
            predict the target
    :param list models_nr: A list of indexes that will be used to point to the trained models which were saved
            locally after training.
    :param str save_models_dir: The path where the models will be saved.
    :param str model_type: The type of the model that was used to fit the data e.g. linear or lightgbm

    :return:
            - y_prediction - A numpy array that contains the predicted target of the unseen data.
    """

    # Predict the target of unseen data
    y_prediction = 0

    if isinstance(predict, bool):
        logger.info("Nothing will be predicted")
    else:
        try:
            y_prediction = predict_unseen_data(predict, models_nr, save_models_dir, model_type)
            logger.info("Prediction process is finished!")
        except Exception as e:
            logger.error(f"Prediction failed: The error is {e}")

    return y_prediction
