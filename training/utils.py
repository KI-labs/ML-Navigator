import logging
import os
import pickle
import shutil
from typing import Union

import numpy as np
from sklearn.model_selection import train_test_split

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


def read_kfold_config(split: dict):
    """ KFold values reader

    This function ensures that the parameters of the KFold splitting method are defined.

    :param dict split: A dictionary that contains the parameters about the KFold splitting method.

    :return:
            - n_fold - An integer that refers to the number of folds which will be used for cross-validation.
            -  shuffle - A boolean. If true, data will be shuffled before splitting it to multiple folds.
            - random_state - An integer which helps to reproduce the results.

    """

    try:
        n_fold = split["fold_nr"]  # number of folds
    except Exception as e:
        print(f"fold_nr is not provided: {e}")
        n_fold = 5

    try:
        shuffle = split["shuffle"]
    except Exception as e:
        print(f"shuffle is not provided: {e}")
        shuffle = True

    try:
        random_state = split["random_state"]
    except Exception as e:
        print(f"random_state is not provided: {e}")
        random_state = 1

    return n_fold, shuffle, random_state


def create_model_directory(path: str):
    """ Model directory creator

    This function create a directory where the model during and after training will be saved.

    :param str path: It refers to the location where the models should be saved.

    """

    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"{path} did not exist. New directory was created")
        else:
            shutil.rmtree(path)  # removes all the subdirectories!
            os.makedirs(path)
            logger.info(f"{path} existed. It was removed and new directory was created")
    except Exception as e:
        print(f"Creating models directory failed: {e}")
        logger.error(f"Creating models directory failed. Error is: {e}")


def save_model_locally(path: str, model: object):
    """ Model saver

    This function saves the model locally in pickle format.

    :param str path: It refers to the location where the models should be saved.
    :param object model: An object created by the training package e.g. Scikit Learn.

    """

    try:
        with open(path, "wb") as save_model:
            pickle.dump(model, save_model)
            logger.info(f"model was saved to {path}")
    except Exception as e:
        print(f"Saving model failed {e}")
        logger.error(f"Saving model failed. Error is: {e}")


def input_parameters_extraction(parameters: dict):
    """ Input data parsing

    :param parameters: dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations. Check the example below.
    :return:
            - data - A dictionary that contains pandas dataframes as datasets.
            - split - A dictionary that contains information about the cross-validation method.
            - train_array - A numpy array that is used to train the model and predict the target.
            - target - A numpy array that is used to train the model.
            - predict - If provided, a pandas dataframe that contains the features without the labels (target). Otherwise bool: False
    """

    try:
        data: dict = parameters["data"]
        split: dict = parameters["split"]

        train_array = data["train"]["features"].values
        target = np.array(data["train"]["target"])
        model_type = parameters["model"]["type"]
        hyperparameters = parameters["model"]["hyperparameters"]

        if "predict" in parameters.keys():
            predict: dict = parameters["predict"]
            logger.info("predict key is given")
        else:
            predict = False
            logger.info("predict key was not found")

        logger.info("Extracting the variables values is finished")
        return data, split, train_array, target, model_type, hyperparameters, predict

    except Exception as e:
        print(f"Parameters extraction failed: {e}")
        logger.error(f"Parameters extraction failed: {e}")


def split_dataset(features: np.array, target: np.array, tests_split_ratios: Union[list, set],
                  stratify: bool = False) -> dict:
    """ Dataset splitter

    This function split a dataset to multiple datasets such train, valid and test.

    :param np.array features: The original dataset that should be split to subsets
    :param np.array target: The original target/labels dataset that should be predicted
    :param Union[list, set] tests_split_ratios: A list or set of floats that represent the ratio of the size of the
            test dataset ot the train dataset. The values should be in the range ]0, 1[ e.g.
            tests_split_ratios = [0.2, 0.2]
    :param bool stratify: If set to True the ratios of the labels is kept the same in the splitted data sets.
    :return:
            sub_datasets: A dictionary that contains the test and train dataset

    :rtype: dict
    """

    # A counter for storing sub-datasets inside the dictionary
    split_count = 0
    sub_datasets = {}

    for split_ratio in tests_split_ratios:
        stratify_ = target if stratify else None
        features, x_test, target, y_test = train_test_split(features,
                                                            target,
                                                            test_size=split_ratio,
                                                            stratify=stratify_,
                                                            random_state=42)

        sub_datasets[f"features_{split_count}"] = x_test
        sub_datasets[f"target_{split_count}"] = y_test
        split_count += 1

    # The remaining dataset after having n split. This dataset will be used to train the model
    sub_datasets["train"] = features
    sub_datasets["target"] = target

    number_of_datasets = len(tests_split_ratios) + 1
    logger.info(f"Splitting the dataset to {number_of_datasets} sub-datasets is finished")
    return sub_datasets


def xgboost_problem_type(hyperparameters: dict) -> str:
    """ XGboost problem type finder

    The function finds the type of the problem that should be solved when using the xgboost method. The function uses
     the objective variable to find out what type of problem that should be solved

    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :return:
            - problem_to_solve: A string that defines the problem to solve: regression or classification.
    """

    try:
        # the default value of the objective is regression reg:squarederror.
        # Here the exception is caught to avoid that the objective is not given by the user
        objective = hyperparameters["objective"]
        if objective.startswith("reg"):
            problem_to_solve = "regression"
        else:
            problem_to_solve = "classification"
    except Exception as e:
        print(f"The objective is not defined. The default value is reg:squarederror. Error: {e}")
        problem_to_solve = "regression"
    return problem_to_solve


def define_model_directory_name(model_type: str, hyperparameters: dict, split: str, problem_to_solve: str):
    """ Assign directory name

    The function defines the name of the directory where the models will be saved locally after training.
    The name contains the information about the model type, split method and some information from the hyperparameters

    :param str model_type: The type of model that will be used to fit the data.
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :param str split: A string that define the data splitting method: split or kfold
    :param str problem_to_solve: A string that defines the problem to solve: regression or classification.
    :return:
            - save_models_dir: The name of the directory where the trained models are saved locally
    """

    # define the name of the directory where the models will be saved
    if model_type == "Ridge linear regression":
        alpha = hyperparameters["alpha"]
        save_models_dir = os.path.join(".", "models", f'linear_ridge_{split}_{alpha}')
    elif model_type == "lightgbm":
        save_models_dir = os.path.join(
            ".", "models", f'lightgbm_{split}_{hyperparameters["num_leaves"]}' +
                           f'_{hyperparameters["boosting"]}')
    elif model_type == "Logistic regression":
        save_models_dir = os.path.join(".", "models", f'logistic_{split}')
    elif model_type == "xgboost":
        save_models_dir = os.path.join(".", "models", f'xgboost_{split}')
    elif model_type.split(".")[0] == "sklearn":
        save_models_dir = os.path.join(".", "models", model_type)
    else:
        raise ValueError("Model type is not recognized")

    save_models_dir += "_" + problem_to_solve

    return save_models_dir
