import collections
import logging
import os

import numpy as np
import pandas as pd

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


def _raise_metrics_error():
    raise ValueError("Invalid parameters structure. The metrics should be a list of strings that have maximum two "
                     "elements: r2_score and mean_squared_error for regression, and accuracy_score and"
                     " roc_auc_score for classification")


class StructureValidation:
    """ Parameters structure validation

    Validate the input parameters and raise an Exception if the structure of the parameters is invalid

    :param dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations.

    :raise:
            | ValueError:
            | TypeError:
    """

    def __init__(self, parameters):
        """
        :param dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations.
        """

        self.parameters = parameters

    def validate_main_keys(self):
        """Main keys validator

        The user has to provide at least those four keys inside the parameters dictionary.

        """

        for key_i in ["data", "split", "model", "metrics"]:
            if key_i not in self.parameters.keys():
                raise ValueError(f"Invalid parameters structure. The key {key_i} is missing")

    def validate_data(self):
        """data key validator

        The following assumptions have to be met:\n
        1. At least there is one dataset inside the data and it should had the key name: train\n
        2. All provided datasets should have two elements: features and target\n
        3. The value of the key features is a pandas dataframe. The target/label should not be between the features\n
        4. The target is a numpy array

        """

        if "train" not in self.parameters["data"].keys():
            raise ValueError("Invalid parameters structure. The key `train` inside `data` is missing")

        for key_i in self.parameters["data"].keys():
            if "features" not in self.parameters["data"][key_i].keys():
                raise ValueError(f"Invalid parameters structure. The key `features` inside the key {key_i} inside "
                                 "`data` is missing")
            if "target" not in self.parameters["data"][key_i].keys():
                raise ValueError(f"Invalid parameters structure. The key `target` inside the key {key_i} inside "
                                 "`data` is missing")
            if not isinstance(self.parameters["data"][key_i]["features"], pd.DataFrame):
                raise TypeError(f"Invalid parameters structure. {key_i}: The features should be a pandas dataframe. "
                                f"The target/labels should not be included in the feature")
            if not isinstance(self.parameters["data"][key_i]["target"], np.ndarray):
                raise TypeError(f"Invalid parameters structure. {key_i}: The target should be a numpy array.")

    def validate_split(self):
        """split key validator

        The following assumptions should be met:\n
        1. There are two elements inside the `split` key: `method` and `split_ratios` or `fold_nr`\n
        2. If the value of the `method` element is `split`, the second element should be `split_ratios`\n
        3. If the value of the `method` element is `kfold`, the second element should be `fold_nr`\n
        4. The `split_ratios` can be either a float or set/list of two floats.\n
        5. The `split_ratios` values should be in ]0, 1[\n
        6. The value of the `fold_nr` should be an integer larger than 1\n
        7. The `method` can take only two values: split` or `kfold`

        """

        if "method" not in self.parameters["split"].keys():
            raise ValueError("Invalid parameters structure. The key `method` inside the key split is missing")

        if self.parameters["split"]["method"] == "split":
            if "split_ratios" not in self.parameters["split"].keys():
                raise ValueError("Invalid parameters structure. The key `split_ratios` inside the key split is missing")

            elif isinstance(self.parameters["split"]["split_ratios"], float) or \
                    isinstance(self.parameters["split"]["split_ratios"], int):
                if self.parameters["split"]["split_ratios"] <= 0 or \
                        self.parameters["split"]["split_ratios"] >= 1:
                    raise ValueError("Invalid parameters structure. split_ratios should be in ]0, 1[")

            elif isinstance(self.parameters["split"]["split_ratios"], collections.Iterable) and \
                    ((len(self.parameters["split"]["split_ratios"]) != 2) |
                     (min(self.parameters["split"]["split_ratios"]) <= 0) |
                     (max(self.parameters["split"]["split_ratios"]) >= 1)):
                raise ValueError("Invalid parameters structure. split_ratios either float or a list or a set that has "
                                 "two floats in the range of ]0, 1[")

        elif self.parameters["split"]["method"] == "kfold":
            if "fold_nr" not in self.parameters["split"].keys():
                raise TypeError("Invalid parameters structure. The key `fold_nr` inside the key split is missing")
            elif not isinstance(self.parameters["split"]["fold_nr"], int) or \
                    self.parameters["split"]["fold_nr"] < 1:
                raise ValueError("Invalid parameters structure. The key `fold_nr` should be an integer larger than 1")

        if self.parameters["split"]["method"] != "split" and self.parameters["split"]["method"] != "kfold":
            raise ValueError("Invalid parameters structure. The split method should be either split or kfold ")

    def validate_model(self):
        """ model key validator

        The following assumptions should be met:\n
        1. The elements `type` and `hyperparameters` should be found inside the values of the key `model`.\n
        2. The type of the value of the `hyperparameters` should be a dictionary.

        """

        if "type" not in self.parameters["model"].keys():
            raise TypeError("Invalid parameters structure. The key type inside the key model is missing")
        elif "hyperparameters" not in self.parameters["model"].keys():
            raise TypeError("Invalid parameters structure. The key hyperparameters inside the key model is missing")
        if not isinstance(self.parameters["model"]["hyperparameters"], dict):
            raise TypeError("Invalid parameters structure. Invalid hyperparameters structure")

    def validate_metrics(self):
        """metrics key validator

        The following assumptions should be met:\n
        1. The value of the `metrics` should be a list.\n
        2. Currently, there are only two regression metrics: r2_score and mean_squared_error, and two classification metrics: accuracy_score and roc_auc_score

        """

        if isinstance(self.parameters["metrics"], list):
            if len(self.parameters["metrics"]) > 2 or len(self.parameters["metrics"]) == 0:
                _raise_metrics_error()

            else:
                for metric_i in self.parameters["metrics"]:
                    if metric_i not in ["r2_score",
                                        "mean_squared_error",
                                        "accuracy_score",
                                        "roc_auc_score"]:
                        _raise_metrics_error()

        else:
            _raise_metrics_error()

    def validate_predict(self):
        """predict key validator

        The following assumptions should be met:\n
        1. If the predict key exists, all of the datasets should have the key `features`\n
        2. The value of the `features` is a pandas dataframe\n
        3. The datasets inside the key `predict` have no target or labels. It is required to predict the target for
        those datasets.

        """

        if "predict" in self.parameters.keys():
            for key_i in self.parameters["predict"]:
                if "features" not in self.parameters["predict"][key_i].keys():
                    raise ValueError(f"Invalid parameters structure. No features were defined for the dataset {key_i}")
                if not isinstance(self.parameters["predict"][key_i]["features"], pd.DataFrame):
                    raise TypeError(f"Invalid parameters structure. {key_i}: The target should be a pandas dataframe.")

    def features_validator(self):
        """Features validator

        All datasets given inside the parameters object should have the same features.

        :return:
        """

        # list of features inside the train dataset
        train_features = list(self.parameters["data"]["train"]["features"].columns)
        for key_i in self.parameters["data"].keys():
            if train_features != list(self.parameters["data"][key_i]["features"].columns):
                raise ValueError(
                    f"Invalid parameters structure. The features of the given dataset data[{key_i}] are not "
                    f"identical with the features of the train dataset")

        if "predict" in self.parameters.keys():
            for key_i in self.parameters["predict"].keys():
                if train_features != list(self.parameters["predict"][key_i]["features"].columns):
                    raise ValueError(
                        f"Invalid parameters structure. The features of the given dataset predict[{key_i}] are not "
                        f"identical with the features of the train dataset")


def parameters_validator(parameters):
    """ Parameters structure validator

    Apply all validation methods defined inside the class `StructureValidation`

    :param dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations.

    """

    validator = StructureValidation(parameters)
    validator.validate_main_keys()
    logger.info("All main keys are there")

    validator.validate_data()
    logger.info("Validating data is finished")

    validator.validate_split()
    logger.info("Validating the split data is finished")

    validator.validate_metrics()
    logger.info("Validating the metrics is finished")

    validator.validate_model()
    logger.info("Validating the model data is finished")

    validator.validate_predict()
    logger.info("Validating the predict is finished")

    validator.features_validator()
    logger.info("Validating the features is finished")

    return True
