import logging
import os
import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from IPython.display import display

from training.model_evaluator import evaluate_model, load_all_models
from training.optimizer import get_best_alpha_split, get_best_alpha_kfold
from training.utils import input_parameters_extraction, define_model_directory_name
from training.utils import read_kfold_config, create_model_directory, \
    save_model_locally, split_dataset, xgboost_problem_type
from training.validator import parameters_validator
from training.xgboost_train import training_xgboost_n_split, \
    training_xgboost_kfold, get_num_round, xgboost_data_preparation_for_evaluation
from training.gridsearch_train import train_sklearn_grid_search

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


def train_with_n_split(test_split_ratios: list, stratify: bool,
                       hyperparameters: dict,
                       train_array: np.array,
                       target: np.array,
                       models_nr: list,
                       model_type: str,
                       required_metrics: list):
    """ n split training

    This function trains a model to fit the data using n split cross-validation e.g train, test or train, valid and test

    :param list test_split_ratios: A list that contains the test split ratio e.g. [0.2] for testing size/training size
            or [0.2, 0.2] for validation size/training size and testing size/(training size - validation size)
    :param bool stratify: If set to True the ratios of the labels is kept the same in the splitted data sets.
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :param np.array train_array: The values of the features that will be split to two sub-datasets based
            on the split value to multiple datasets.
    :param np.array target: The values of the target that will be split to two sub-datasets based
            on the split value to multiple datasets.
    :param list models_nr: A list of indexes that will be used to point to the trained models which will be saved
            locally after training. In this case there is only one model.
    :param str model_type: The type of model that will be used to fit the data. Currently there are two values:
            Ridge linear regression and lightgbm.
    :param list required_metrics:

    :return:
            - models_nr -  A list of indexes that will be used to point to the trained models which will be saved locally after training. In this case there is only one model.
            - save_models_dir - The name of the directory where the trained models are saved locally.
    """

    problem_to_solve = None
    sub_datasets = split_dataset(train_array, target, test_split_ratios, stratify)

    x_train, y_train = sub_datasets["train"], sub_datasets["target"]

    if model_type == "Ridge linear regression":

        problem_to_solve = "regression"

        logger.info("Ridge linear regression will be used to train the model")
        alpha = hyperparameters["alpha"]

        if alpha == "optimize":
            best_alpha = get_best_alpha_split(x_train, y_train,
                                              sub_datasets["features_0"],
                                              sub_datasets["target_0"])
            linear_regression_model = linear_model.Ridge(alpha=best_alpha)
            print("the optimized alpha value", best_alpha)
        else:
            linear_regression_model = linear_model.Ridge(alpha=alpha)

        model = linear_regression_model.fit(x_train, y_train)

    elif model_type == "lightgbm":

        if hyperparameters["objective"] != "regression":
            problem_to_solve = "classification"
        else:
            problem_to_solve = "regression"

        logger.info("LightGBM regression will be used to train the model")
        lgb_train = lgb.Dataset(x_train, label=y_train.reshape(len(y_train)))

        lgb_valid = lgb.Dataset(sub_datasets["features_0"],
                                label=sub_datasets["target_0"].reshape(len(sub_datasets["target_0"])))

        # start training the model
        model = lgb.train(hyperparameters, train_set=lgb_train, valid_sets=[lgb_valid], verbose_eval=0)

    elif model_type == "Logistic regression":
        problem_to_solve = "classification"
        logger.info("Logistic regression will be used to train the model")
        if len(set(y_train)) > 2:
            logistic_regression_model = LogisticRegression(multi_class='multinomial')
        else:
            logistic_regression_model = LogisticRegression()

        model = logistic_regression_model.fit(x_train, y_train)

    elif model_type == "xgboost":
        num_round = get_num_round(hyperparameters)
        model, problem_to_solve, validation_list = training_xgboost_n_split(sub_datasets, hyperparameters, num_round)

    elif model_type.split(".")[0] == "sklearn":
        model = train_sklearn_grid_search(x_train, y_train, model_type, hyperparameters)
        problem_to_solve = hyperparameters["objective"]
    else:
        logger.error("Model type is not recognized")
        raise ValueError("Model type is not recognized")

    # Evaluate the model on all sub-datasets
    metrics_summary_all = dict()
    metrics_summary_all["model 0"] = evaluate_model(
        model,
        xs=[x_train] + [sub_datasets[f"features_{i}"] for i in range(len(test_split_ratios))],
        ys=[y_train] + [sub_datasets[f"target_{i}"] for i in range(len(test_split_ratios))],
        labels=["(train.train)"] + [f"(train.validation_{i})" for i in range(len(test_split_ratios))],
        metrics=required_metrics)

    metrics_summary_all = pd.DataFrame(metrics_summary_all)
    display(metrics_summary_all)

    # Define the directory name of the models
    split = "n_split"
    save_models_dir = define_model_directory_name(model_type, hyperparameters, split, problem_to_solve)
    # Get a unique name when saving the model
    for n_split in test_split_ratios:
        save_models_dir += f'_{str(n_split).replace(".", "_")}'
    logger.debug(f"the name of the directory where the model will be saved is: {save_models_dir}")
    create_model_directory(save_models_dir)

    path = os.path.join(save_models_dir, f"{model_type}_{0}.pkl")
    save_model_locally(path, model)

    # we have only one model in this case
    models_nr.append("0")

    logger.info("Training the model using split cross-validation is finished")
    return models_nr, save_models_dir


def train_with_kfold_cross_validation(split: dict, stratify: bool,
                                      hyperparameters: dict,
                                      train_array: np.array,
                                      target: np.array,
                                      models_nr: list,
                                      model_type,
                                      required_metrics: list):
    """K-Fold cross-validation training

    This function trains a model to fit the data using K-Fold cross-validation.

    :param dict split: A dictionary that contains information about the K-Fold variables
    :param bool stratify: If set to True the ratios of the labels is kept the same in the splitted data sets.
    :param dict hyperparameters: A dictionary that contains the hyperparameters which the selected training method
            needs to train the model.
    :param np.array train_array: The values of the target that will be split into K-Folds and used to train the model to
            predict the target
    :param np.array target: The values of the target that will be split into K-Folds and used to train the model.
    :param list models_nr: A list of indexes that will be used to point to the trained models which will be saved
            locally after training. In this case there are n_fold models.
    :param str model_type: The type of model that will be used to fit the data.
    :param list required_metrics:

    :return:
            - models_nr - A list of indexes that will be used to point to the trained models which will be saved locally after training. In this case there are n_fold models.
            - save_models_dir - The name of the directory where the trained models are saved locally.
    """
    problem_to_solve = None

    # Ensure that the user provided the required variable.
    n_fold, shuffle, random_state = read_kfold_config(split)

    if stratify:
        kfold = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)
    else:
        kfold = KFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)

    # Avoiding warning of local variable might be referenced before assignment
    linear_regression_model = None
    logistic_regression_model = None

    if model_type == "Ridge linear regression":
        problem_to_solve = "regression"
        alpha = hyperparameters["alpha"]
        if alpha == "optimize":
            alpha = get_best_alpha_kfold(kfold, train_array, target)
        linear_regression_model = linear_model.Ridge(alpha=alpha)
    elif model_type == "lightgbm":
        if hyperparameters["objective"] != "regression":
            problem_to_solve = "classification"
        else:
            problem_to_solve = "regression"
    elif model_type == "Logistic regression":
        problem_to_solve = "classification"
        if len(set(target)) > 2:
            logistic_regression_model = LogisticRegression(multi_class='multinomial')
        else:
            logistic_regression_model = LogisticRegression()
    elif model_type == "xgboost":
        # Find out what is the type of the problem that should be solved based on the defined objective
        problem_to_solve = xgboost_problem_type(hyperparameters)
    elif model_type.split(".")[0] == "sklearn":
        problem_to_solve = hyperparameters["objective"]

    fold_nr = 0  # counter for identifying models
    metrics_summary_all = {}
    # Define the directory name of the models
    split = "kfold"
    save_models_dir = define_model_directory_name(model_type, hyperparameters, split, problem_to_solve)
    save_models_dir += f"_folds_n_{n_fold}"
    create_model_directory(save_models_dir)

    for train, test in kfold.split(train_array, target):
        fold_nr += 1
        print("fold_nr.", fold_nr)

        if model_type == "Ridge linear regression":
            logger.info("LightGBM regression will be used to train the model")
            kfold_model = linear_regression_model.fit(train_array[train], target[train])

        elif model_type == "lightgbm":
            logger.info("LightGBM will be used to train the model")
            lgb_train = lgb.Dataset(train_array[train], label=target[train].reshape(len(target[train])))
            lgb_valid = lgb.Dataset(train_array[test], label=target[test].reshape(len(target[test])))

            # start training the model
            kfold_model = lgb.train(hyperparameters, train_set=lgb_train, valid_sets=[lgb_valid], verbose_eval=0)

        elif model_type == "Logistic regression":
            kfold_model = logistic_regression_model.fit(train_array[train], target[train])

        elif model_type == "xgboost":
            num_round = get_num_round(hyperparameters)
            kfold_model, validation_list = training_xgboost_kfold(train_array,
                                                                  target,
                                                                  train,
                                                                  test,
                                                                  hyperparameters,
                                                                  num_round)

        else:
            logger.error("Model type is not recognized")
            raise ValueError("Model type is not recognized")

        path = os.path.join(save_models_dir, f"{model_type}_{fold_nr}.pkl")
        save_model_locally(path, kfold_model)

        models_nr.append(fold_nr)

        if model_type == "xgboost":
            metrics_summary = evaluate_model(kfold_model,
                                             xs=[validation_list[0][0], validation_list[1][0]],
                                             ys=[target[train], target[test]],
                                             labels=["(train.train)", "(train.validation)"],
                                             metrics=required_metrics)
        else:
            metrics_summary = evaluate_model(kfold_model,
                                             xs=[train_array[train], train_array[test]],
                                             ys=[target[train], target[test]],
                                             labels=["(train.train)", "(train.validation)"],
                                             metrics=required_metrics)

        metrics_summary_all[f"fold_{fold_nr}"] = metrics_summary
        print(metrics_summary)

    metrics_summary_all = pd.DataFrame(metrics_summary_all)
    metrics_summary_all['mean'] = metrics_summary_all.mean(axis=1)
    display(metrics_summary_all)

    logger.info("Training the model using KFold cross-validation is finished")
    return models_nr, save_models_dir


def model_training(parameters: dict):
    """ Model training

    This function trains a model to fit the data using the Scikit Learn of Ridge linear model implementation

    :param dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations. Check the example below.
    :return:
            - models_nr - A list of indexes that will be used to point to the trained models which will be saved locally after training.
            - save_models_dir - The name of the directory where the trained models are saved locally.
    :example:
            One split: train and test
                    >>> parameters = {
                    >>>      "data": {
                    >>>          "train": {"features": train_dataframe, "target": train_target},
                    >>>          "valid": {"features": valid_dataframe, "target": valid_target}, # optional
                    >>>          "test": {"features": test_dataframe, "target": test_target}, # optional
                    >>>      },
                    >>>      "split": {
                    >>>          "method": "split",
                    >>>          "split_ratios": 0.2,
                    >>>      },
                    >>>      "model": {"type": "Ridge linear regression",
                    >>>                "hyperparameters": {"alpha": 1,
                    >>>                                    },
                    >>>                },
                    >>>      "metrics": ["r2_score", "mean_squared_error"],
                    >>>      "predict": { # optional
                    >>>          "test": {"features": test_dataframe}
                    >>>      }
                    >>>  }
            Two splits: train, valid and test
                    >>> parameters = {
                    >>>      "data": {
                    >>>          "train": {"features": train_dataframe, "target": train_target},
                    >>>          "valid": {"features": valid_dataframe, "target": valid_target}, # optional
                    >>>          "test": {"features": test_dataframe, "target": test_target}, # optional
                    >>>      },
                    >>>      "split": {
                    >>>          "method": "split",
                    >>>          "split_ratios": (0.2, 0.2), # or [0.2, 0.2]
                    >>>      },
                    >>>      "model": {"type": "Ridge linear regression",
                    >>>                "hyperparameters": {"alpha": 1,
                    >>>                                    },
                    >>>                },
                    >>>      "metrics": ["r2_score", "mean_squared_error"],
                    >>>      "predict": { # optional
                    >>>          "test": {"features": test_dataframe}
                    >>>      }
                    >>>  }
            KFold cross-validation:
                    >>> parameters = {
                    >>>      "data": {
                    >>>          "train": {"features": train_dataframe, "target": train_target},
                    >>>          "valid": {"features": valid_dataframe, "target": valid_target}, # optional
                    >>>          "test": {"features": test_dataframe, "target": test_target}, # optional
                    >>>      },
                    >>>      "split": {
                    >>>          "method": "kfold",
                    >>>          "fold_nr": 5,
                    >>>      },
                    >>>      "model": {"type": "Ridge linear regression",
                    >>>                "hyperparameters": {"alpha": 1,
                    >>>                                    },
                    >>>                },
                    >>>      "metrics": ["r2_score", "mean_squared_error"],
                    >>>      "predict": { # optional
                    >>>          "test": {"features": test_dataframe}
                    >>>      }
                    >>>  }
            KFold cross-validation with alpha optimization:
                    >>> parameters = {
                    >>>      "data": {
                    >>>          "train": {"features": train_dataframe, "target": train_target},
                    >>>          "valid": {"features": valid_dataframe, "target": valid_target}, # optional
                    >>>          "test": {"features": test_dataframe, "target": test_target}, # optional
                    >>>      },
                    >>>      "split": {
                    >>>          "method": "kfold",
                    >>>          "fold_nr": 5,
                    >>>      },
                    >>>      "model": {"type": "Ridge linear regression",
                    >>>                "hyperparameters": {"alpha": "optimize",
                    >>>                                    },
                    >>>                },
                    >>>      "metrics": ["r2_score", "mean_squared_error"],
                    >>>      "predict": { # optional
                    >>>          "test": {"features": test_dataframe}
                    >>>      }
                    >>>  }

    """

    parameters_validator(parameters)
    data, split, train_array, target, model_type, hyperparameters, predict = input_parameters_extraction(parameters)
    stratify = parameters["split"].get("stratify", False)

    models_nr = []
    save_models_dir = None

    if split["method"] == "split":
        logger.info("Start training using Split cross-validation")
        if type(split["split_ratios"]) == float:
            test_split_ratio = [split["split_ratios"]]
        else:
            test_split_ratio = split["split_ratios"]

        models_nr, save_models_dir = train_with_n_split(test_split_ratio,
                                                        stratify,
                                                        hyperparameters,
                                                        train_array,
                                                        target,
                                                        models_nr,
                                                        model_type,
                                                        parameters["metrics"])

    elif split["method"] == "kfold":
        logger.info("Start training using KFold cross-validation")
        models_nr, save_models_dir = train_with_kfold_cross_validation(split,
                                                                       stratify,
                                                                       hyperparameters,
                                                                       train_array,
                                                                       target,
                                                                       models_nr,
                                                                       model_type,
                                                                       parameters["metrics"])

    # Evaluate models
    if model_type == "xgboost":
        data = xgboost_data_preparation_for_evaluation(data)

    metrics_summary_all = {}
    for model_i in models_nr:
        model = load_all_models(save_models_dir, model_type, model_i)
        xs = [v["features"] for k, v in data.items()]
        ys = [v["target"] for k, v in data.items()]
        labels = [f"({k})" for k, v in data.items()]

        metrics_summary = evaluate_model(model, xs, ys, labels, parameters["metrics"])

        metrics_summary_all[f"model {model_i}"] = metrics_summary

    metrics_summary_all = pd.DataFrame(metrics_summary_all)
    if metrics_summary_all.shape[1] > 1:
        # calculate mean only if we have more than one model
        metrics_summary_all['mean'] = metrics_summary_all.mean(axis=1)
    display(metrics_summary_all)

    logger.info("Training process is finished")
    return models_nr, save_models_dir
