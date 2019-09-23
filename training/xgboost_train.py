import xgboost as xgb
import numpy as np
import pandas as pd
from training.utils import xgboost_problem_type


def xgboost_data_preparation(validation_list: list, dataframe: pd.DataFrame, target: np.array, key: str):
    """ xgboost data preparing for training

    The function transforms the data from a Pandas dataframe format to a xgboost-compatible format.

    :param list validation_list: The list that contains the data the should be used to train and validate the model.
    :param pd.DataFrame dataframe: Pandas dataframe that contains the data which will be transformed to xgboost format.
    :param np.array target: An array that contains the target that should be predict by the xgboost model
    :param str key: A label that is used to name the dataset in the validation_list

    :return:
            - The updated validation_list
    """

    xgboost_data_train = xgb.DMatrix(dataframe, label=target)
    validation_list.append((xgboost_data_train, key))

    return validation_list


def xgboost_regression_train(validation_list: list,
                             hyperparameters: dict,
                             num_round: int = 10):
    """ xgboost trainer

    The function uses the xgboost framework to train the model

    :param list validation_list: The list that contains the data the should be used to train and validate the model.
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :param int num_round: The number of rounds for boosting

    :return:
            - xgboost model
    """

    param = hyperparameters
    bst = xgb.train(param, validation_list[0][0], num_round, validation_list)
    return bst


def xgboost_data_preparation_to_predict(dataframe: pd.DataFrame):
    """ xgboost data preparing for prediction

    The function transforms the data from a Pandas dataframe format to a xgboost-compatible format

    :param pd.DataFrame dataframe: Pandas dataframe that contains the data which will be transformed to xgboost format.

    :return:
            - The dataset in xgboost-compatible format
    """

    data = xgb.DMatrix(dataframe)

    return data


def training_xgboost_n_split(sub_datasets: dict, hyperparameters: dict, num_round: int = 10):
    """ XGboost training with n-split

    This function trains a model to fit the data using n split cross-validation e.g train, test or train, valid and test

    :param dict sub_datasets:
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            need to train the model.
    :param int num_round: The number of rounds for boosting

    :return:
            - model: xgboost model.
            - problem_to_solve: string that defines the problem to solve: regression or classification.
            - validation_list: The list that contains the data the should be used to train and validate the model.
    """

    # Find out what is the type of the problem that should be solved based on the defined objective
    problem_to_solve = xgboost_problem_type(hyperparameters)

    # train data transformation
    validation_list = []
    data_i = "train"
    dataframe = pd.DataFrame(sub_datasets["train"])
    target = sub_datasets["target"]
    validation_list = xgboost_data_preparation(validation_list, dataframe, target, data_i)

    sub_datasets["train"] = validation_list[-1][0]

    # transformation of the reset of the datasets
    for index_i in range(len(list(sub_datasets.keys()))):
        try:
            data_i = f"dataset_{index_i}"
            print(data_i)
            dataframe = pd.DataFrame(sub_datasets[f"features_{index_i}"])
            target = sub_datasets[f"target_{index_i}"]
            validation_list = xgboost_data_preparation(validation_list, dataframe, target, data_i)
            sub_datasets[f"features_{index_i}"] = validation_list[-1][0]
        except:
            print("All dataset are considered")
            break

    model = xgboost_regression_train(validation_list, hyperparameters, num_round=num_round)

    return model, problem_to_solve, validation_list


def training_xgboost_kfold(train_array, target, train: list, test: list, hyperparameters: dict, num_round: int = 10):
    """ XGboost training with kfold

    This function trains a model to fit the data using K-Fold cross-validation.

    :param np.array train_array: The values of the target that will be split into K-Folds and used to train the model to
            predict the target
    :param np.array target: The values of the target that will be split into K-Folds and used to train the model.
    :param list train: A list of integers that define the training dataset
    :param list test: A list of integers that define the testing dataset
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :param int num_round: The number of rounds for boosting

    :return:
            - kfold_model: xgboost model
            - problem_to_solve: string that defines the problem to solve: regression or classification.
            - validation_list: The list that contains the data the should be used to train and validate the model.
    """

    # train data transformation
    validation_list = []
    data_i = "train"
    dataframe = pd.DataFrame(train_array[train])
    xgboost_target = target[train]
    validation_list = xgboost_data_preparation(validation_list, dataframe, xgboost_target, data_i)

    # test data transformation
    data_i = "test"
    dataframe = pd.DataFrame(train_array[test])
    xgboost_target = target[test]
    validation_list = xgboost_data_preparation(validation_list, dataframe, xgboost_target, data_i)

    kfold_model = xgboost_regression_train(validation_list, hyperparameters, num_round=num_round)

    return kfold_model, validation_list


def get_num_round(hyperparameters) -> int:
    """ num_round getter

    Get the value of num_round that will be used to train the xgboost model

    :param hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :return:
            - num_round: The number of rounds for boosting
    """

    try:
        num_round = hyperparameters["num_round"]
    except Exception as e:
        print(f"The num_round is not defined. The default value is num_round = 10. Error: {e}")
        num_round = 10
    return num_round


def xgboost_data_preparation_for_evaluation(data: dict):
    """ Date preparation for evaluation

    Prepare the data in a form that could be used for model evaluation.

    :param data:
    :return:
    """

    for data_i in data.keys():
        dataframe = data[data_i]["features"]
        dataframe.columns = [x for x in range(len(list(dataframe.columns)))]
        target = data[data_i]["target"]

        data[data_i]["features"] = xgb.DMatrix(dataframe, label=target)

    return data
