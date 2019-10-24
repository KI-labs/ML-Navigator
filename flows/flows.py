""" Flows pieces collector

This module contains the Flows class, which is a container of the
methods that construct the flows.

The modules depends on the classes and methods which are writen
in other package in this project.

"""

import json
import logging
import os
from typing import Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import yaml
from blessings import Terminal

from feature_engineering.feature_generator import encoding_features
from flows.flow_instructions import FlowInstructions
from flows.utils import unify_dataframes
from prediction.model_predictor import model_prediction
from preprocessing.data_clean import drop_corr_columns, drop_const_columns
from preprocessing.data_explorer import explore_data
from preprocessing.data_science_help_functions import detect_id_target_problem
from preprocessing.data_transformer import encode_categorical_features, standard_scale_numeric_features, clean_categorical_features
from preprocessing.data_type_detector import detect_columns_types_summary
from preprocessing.json_preprocessor import flat_json
from preprocessing.utils import check_if_target_columns_are_imbalanced
from preprocessing.utils import read_data
from preprocessing.data_explorer import outliers_detector
from training.training import model_training
from visualization.visualization import compare_statistics

logger = logging.getLogger(__name__)
formatting = (
    "%(asctime)s: %(levelname)s: File:%(filename)s Function:%(funcName)s Line:%(lineno)d "
    "message:%(message)s"
)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "logs.log"),
    level=logging.DEBUG,
    format=formatting,
)

# For colorful and beautiful formatted print()
term = Terminal()

print(term.bold(term.magenta("Welcome to the Data Science Package. First create an object as follows:")))
print(term.bold(term.magenta("For example, use the code below to import the flow 0:")))

print(term.green_on_black("flow = Flows(0)"))
print(term.bold(term.magenta("You can define the `categorical_threshold` which is the maximum number of categories"
                             " that a categorical feature should have before considering it as continuous"
                             " numeric feature. The default value is 50")))
print(term.bold(term.magenta("For example, use the code below to import the flow 0"
                             " with defining the categorical_threshold as 50")))
print(term.green_on_black("flow = Flows(flow_id=0, categorical_threshold=50)"))


class Flows:
    """
    Flows methods container

    A class which meant to be a container for all flows elements.

    :param int flow_id: An integer which points to the flow that the
            use wants to follow.
    :param int categorical_threshold: The maximum number of categories that a categorical feature should have before
                                      considering it as continuous numeric feature.
    :param object commands: It contains the list of the instructions
            that are loaded from the yaml file
    :param list columns_set: A dictionary that contains the features'
                                names sorted in multiple lists based on the
                                type of the data for each given dataset.

    :methods:
            - `guidance` - Evaluate YAML commands
            - `load_data` - Read CSV data
            - `encode_categorical_feature` - Encode the categorical features by changing the string value to numeric values
            - `scale_data` - Scale the numerical value (Feature Standardization - mean = 0, STD = 1)
            - `one_hot_encoding` - Encode categorical features using one-hot encoding method
            - `training_ridge_linear_model` - Train a model using the regression Scikit Learn Ridge linear model implementation
            - `training_lightgbm` - Train a tree-based regression model using the LightGBM implementation

    """

    def __init__(self, flow_id: int, categorical_threshold: int = 50, kl_div_threshold: float = 0.05):
        """

        :param int flow_id: An integer which points to the flow that the use wants to follow.
        :param int categorical_threshold: The maximum number of categories that a categorical feature should have before
                                      considering it as continuous numeric feature.

        """

        self.columns_set = None
        self.flow_id = flow_id
        self.flow_steps = {}
        self.categorical_threshold = categorical_threshold
        self.kl_div_threshold = kl_div_threshold

        # load the yaml file that contains the instructions of the flow
        flow_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flows", f"flow_{self.flow_id}.json")
        flow_instruction_database = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                 "flows", "flow_instructions_database.yaml")
        try:
            with open(flow_instruction_database, 'r') as stream:
                self.commands = yaml.safe_load(stream)
                logger.info("flow_instruction_database.yaml was loaded successfully.")
        except yaml.YAMLError as exc:
            logger.error(f"Loading the yaml file failed. The error is {exc}")

        try:
            with open(flow_path, 'r') as stream:
                self.flow_steps = json.load(stream)
                logger.info(f"flow_{self.flow_id}.json was loaded successfully.")
        except Exception as exc:
            logger.error(f"Loading the json file failed. The error is {exc}")

        # First thing first: tell the user how to read the data and explore the features
        FlowInstructions.read_data()
        FlowInstructions.explore_data()

    def guidance(self, step_ext: object):
        """ YAML evaluator

        This function executes the command that is written in the yaml file under a certain step

        :param object step_ext: It can be an integer that points to a certain step e.g. 1 or a combination of both
                an integer and a letter to point to a sub-step e.g. 1_a

        """

        for command in self.commands[step_ext][0]["guide"]:
            try:
                eval(command)
                logger.info(f"The command in the step {step_ext} was executed successfully")
            except Exception as e:
                logger.error(f"Failed executing the command in the step {step_ext}. Error is: {e}")

    def load_data(self, path: str, files_list: list, rows_amount: int = 0) -> Tuple[Dict, List]:
        """ Data reader

        This function reads data from CSV files and returns a dictionary
        that contains Pandas dataframes e.g. dataframes_dict={"train":
        train_dataframe, "test": test_dataframe}

        After reading the data, the function provides a summary of
        each dataset.

        After presenting the summary, the function tries to detect
        which column may contain the ids and which column can be
        the target (labels). Based on the values of the target,
        the function can tell if the problem which should be solved
        is a regression or classification problem.

        :param str path: The path to the data.
        :param list files_list: A list of strings which are the
                names of the files
        :param int rows_amount: The number of rows that should be read from the CSV file. If 0, all rows will be read.

        :return:
                - dataframes_dict - A dictionary that contains Pandas dataframes.
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.

        :example:

        >>> path = "./data"
        >>> files_list = ['train.csv','test.csv']

        >>> dataframes_dict={"train": train_dataframe, "test": test_dataframe}
        """

        function_id = "0"

        dataframes_dict = read_data(path, files_list, rows_amount)

        self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)

        print(term.bold(term.magenta("If you are interested in extracting the outliers,"
                                     " you can run the following commands:")))
        print(term.green_on_black("average_score, is_outlier = flow.outliers_extractor"
                                  "(dataframe_dict: dict, key_i: str)"))

        _, _, possible_problems = detect_id_target_problem(dataframes_dict)

        _ = check_if_target_columns_are_imbalanced(dataframes_dict, possible_problems, self.kl_div_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict, self.columns_set

    def encode_categorical_feature(self, dataframes_dict: dict, ignore_columns: list = None,
                                   print_results: Union[bool, int] = False,
                                   _reference: Union[bool, str] = False,
                                   clean_categorical_values: bool = True) -> Tuple[Dict, List]:
        """ Categorical features encoder

        This function encodes the categorical features by replacing
        the strings with integers

        :param Union[bool, int] print_results: If False, no data is printed to the console. If True, all data is printed to the console. If an integer n, only the data for n features is printed to the console.
        :param list ignore_columns: It contains the columns that should be
                ignored when apply scaling e.g. the id and the target.
        :param dict dataframes_dict: A dictionary that contains Pandas dataframes
                before encoding the features e.g. dataframes_dict={"train": train_dataframe, "test": test_dataframe}
        :param Union[bool, str] _reference: The reference dataframe which is used when applying functions to other dataframes. The default value is the first dataframe inside the dataframes_dict dictionary. Usually it is the train dataframe
        :param bool clean_categorical_values: If True (default), non-common cathegorical values will be filtered out (set to NAN) before encoding.

        :return:
                - dataframes_dict_encoded - A dictionary that contains Pandas dataframes after encoding the features e.g. dataframes_dict={"train": train_dataframe, "test": test_dataframe}
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.

        """

        function_id = "1"

        if not _reference:
            _reference = next(iter(dataframes_dict.keys()))

        print(f"The reference dataframe is: {_reference}")

        string_columns = self.columns_set[_reference]["categorical_string"] + self.columns_set[_reference][
            "categorical_integer"]

        if ignore_columns is not None:
            string_columns = [col_i for col_i in string_columns if col_i not in ignore_columns]

        # clean_categorical_values in dataframes
        if clean_categorical_values:
            dataframes_dict = clean_categorical_features(dataframes_dict, string_columns, False)

        # Feature encoding
        dataframes_dict_encoded = encode_categorical_features(dataframes_dict, string_columns, print_results)

        print(term.red("*" * 30))

        # After encoding features, A new summary of the data will be presented
        self.columns_set = detect_columns_types_summary(dataframes_dict_encoded, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict_encoded, self.columns_set

    def scale_data(self, dataframes_dict: dict, ignore_columns: list,
                   _reference: Union[bool, str] = False) -> Tuple[Dict, List]:
        """ Feature scaling

        This function scales features that contains numeric continuous values.

        :param dict dataframes_dict: A dictionary that contains Pandas dataframes
                before scaling features e.g. dataframes_dict={"train": train_dataframe,
                "test": test_dataframe}
        :param list ignore_columns: It contains the columns that should be
                ignored when apply scaling e.g. the id and the target.
        :param Union[bool, str] _reference: The reference dataframe which is used when applying functions to other dataframes. The default value is the first dataframe inside the dataframes_dict dictionary. Usually it is the train dataframe

        :return:
                - dataframes_dict_scaled - A dictionary that contains Pandas dataframes after scaling features e.g. dataframes_dict={"train": train_dataframe, "test": test_dataframe}
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.
        """

        function_id = "2"

        if not _reference:
            _reference = next(iter(dataframes_dict.keys()))

        numerical = [x for x in self.columns_set[_reference]["continuous"] if x not in ignore_columns]

        dataframes_dict_scaled = standard_scale_numeric_features(dataframes_dict, _reference, numerical)

        # Suggesting the next step
        self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict_scaled, self.columns_set

    def features_encoding(self, encoding_type: str, dataframe_dict: dict, reference: str,
                          ignore_columns: list, class_number_range: list = None, target_name: str = None,
                          drop_encoded_features: bool = True):
        """ The encoder

        This function encodes the categorical features using different encoding methods
        It assumes that the user encoded the categorical features with string values
        by replacing the those string values by integers

        :param str encoding_type: The type of the encoding method that will be applied. For example: one-hot, target \n
                    For more information please check the following reference:\n
                    https://contrib.scikit-learn.org/categorical-encoding/index.html
        :param dict dataframe_dict: A dictionary that contains Pandas dataframes before the encoding features e.g.
                dataframes_dict={"train": train_dataframe,
                "test": test_dataframe}.
        :param str reference: It is the key ind the dataframes dictionary that points to the dataframe which its
                inputs are taken as a reference to encode the data of other dataframes e.g. "train".
        :param list ignore_columns: It is a list of strings that contains the name of the columns which should be
                ignored when applying the encoding method.
        :param list class_number_range: It is a list of two elements which define the minimum the and maximum number of
                the classes (unique value) that a feature should contain in order to apply the one-hot encoding to
                this feature.
        :param str target_name: The name of the column that contains the labels that should be predicted by the model.
                            If the encoding method doesn't require that target, it can be ignored.
        :param bool drop_encoded_features: If True, the encoded features will be dropped from the dataset

        :return:
                - dataframe_dict_encoded -  A dictionary that contains Pandas dataframes after the encoding features e.g. dataframe_dict_encoded={"train": train_dataframe, "test": test_dataframe}.
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.
        """

        function_id = f"3-{encoding_type}"

        if class_number_range is None:
            class_number_range = [3, 50]

        categorical_feature = self.columns_set[reference]["categorical_integer"]
        dataframe_dict_encoded = encoding_features(encoding_type,
                                                   dataframe_dict,
                                                   reference,
                                                   categorical_feature,
                                                   ignore_columns,
                                                   class_number_range,
                                                   target_name,
                                                   drop_encoded_features)

        self.columns_set = detect_columns_types_summary(dataframe_dict_encoded, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframe_dict_encoded, self.columns_set

    def training(self, parameters: dict) -> Union[Tuple[List, str], Tuple[List, str, np.array]]:
        """ Ridge linear model

        This function fits the data using the ridge linear model. The function uses the implementation from
        scikit Learn. The user can train a model with specific configuration using the parameter variable.

        :param dict parameters: A dictionary that contains information about the datasets, model type,
                model configurations and training configurations. Check the example below.

        :return:
                - model_index_list - A list of indexes that will be used to point to the trained models which will be saved locally after training
                - save_models_dir - The path where the models will be saved.
                - y_predict - numpy array If the `predict` key is given, the model `my_model` predict the labels of the `parameters["predict"]["test"]` dataset gives back `y_predict`.

        :example:
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
        """

        function_id = "4"

        model_index_list, save_models_dir = model_training(parameters)

        if "predict" in parameters.keys():
            model_type = parameters["model"]["type"]
            predict = parameters["predict"]
            y_predict = model_prediction(predict, model_index_list, save_models_dir, model_type)

            try:
                self.guidance(self.flow_steps[function_id])
            except Exception as e:
                print(f"It seems that the next step is not defined. Error{e}")

            return model_index_list, save_models_dir, y_predict
        else:
            return model_index_list, save_models_dir

    @staticmethod
    def comparing_statistics(dataframe_dict: dict):
        """ Datasets statistics visualizer

        This function visualize the statistical properties of the given datasets. It plots Those properties in a
        single graph which help obtain an overview about the distribution of the data in different datasets. It is an
        interactive function. Therefore, it was designed to run in a Jupyter notebook.

        :param dict dataframe_dict: A dictionary that contains Pandas dataframes  e.g. dataframes_dict={"train":
                train_dataframe, "test": test_dataframe}
        """

        compare_statistics(dataframe_dict)

    @staticmethod
    def exploring_data(dataframe_dict: dict, key_i: str):
        """ Datasets explorer

        This functions explore a given dataset by showing information about the most and the least repeated value,
        the number of unique values and the distribution of each feature. It is an interactive function. Therefore,
        it was designed to run in a Jupyter notebook.

        :param dict dataframe_dict: A dictionary that contains Pandas dataframes  e.g. dataframes_dict={"train":
                train_dataframe, "test": test_dataframe}
        :param str key_i: It points to the dataset which the user wants to explore e.g. "train".
        """

        explore_data(dataframe_dict[key_i])

    def flatten_json_data(self, dataframes_dict: dict, _reference: Union[bool, str] = False) -> Tuple[Dict, List]:
        """ JSON data normalizer

        This function normalizes the nested JSON data type inside the pandas dataframes' columns. The name of the new
         columns has the same name of the parent column with a predefined suffix to ensure unique columns' names.

        :param dict dataframes_dict: A dictionary that contains Pandas dataframes with nested JSON data type e.g.
                dataframes_dict={ "train": train_dataframe, "test": test_dataframe}
        :param Union[bool, str] _reference: The reference dataframe which is used when applying functions to other dataframes. The default value is the first dataframe inside the dataframes_dict dictionary. Usually it is the train dataframe


        :return:

            - dataframes_dict - A dictionary that contains Pandas dataframes after flatting the nested JSON data type e.g. dataframes_dict={ "train": train_dataframe, "test": test_dataframe}
            - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.

        """

        function_id = "5"

        if not _reference:
            _reference = next(iter(dataframes_dict.keys()))

        while len(self.columns_set[_reference]["json"]) > 0:
            dataframes_dict = flat_json(dataframes_dict, self.columns_set[_reference]["json"])
            self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict, self.columns_set

    def drop_correlated_columns(self, dataframes_dict: dict, ignore_columns: list, drop_columns: bool = True,
                                print_columns: bool = True, threshold: float = 0.98,
                                _reference: Union[bool, str] = False) -> Tuple[Dict, List]:
        """ Correlation eliminator

        The function drop correlated columns and keep only one of these columns.

        :param dict dataframes_dict: A dictionary that contains Pandas dataframes
                e.g. dataframes_dict={"train": train_dataframe, "test": test_dataframe}
        :param list ignore_columns: It contains the columns that should be ignored e.g. the id and the target.
        :param bool drop_columns: If true, all correlated columns will be dropped but one.
        :param bool print_columns: If True, information about the correlated columns will be printed to the console.
        :param float threshold: A value between 0 and 1. If the correlation between two columns is larger than this.
                value, they are considered highly correlated. If drop_columns is True, one of those columns will be
                dropped. The recommended value of the `threshold` is in [0.7 ... 1].
        :param Union[bool, str] _reference: The reference dataframe which is used when applying functions to other dataframes. The default value is the first dataframe inside the dataframes_dict dictionary. Usually it is the train dataframe


        :return:
                - dataframes_dict - A dictionary that contains Pandas dataframes after dropping correlated columns e.g. dataframes_dict={ "train": train_dataframe, "test": test_dataframe}
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.
        """

        function_id = "6"

        if not _reference:
            _reference = next(iter(dataframes_dict.keys()))
        dataframes_dict[_reference] = drop_corr_columns(dataframes_dict[_reference],
                                                        drop_columns,
                                                        print_columns,
                                                        threshold)
        dataframes_dict = unify_dataframes(dataframes_dict, _reference, ignore_columns)
        self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict, self.columns_set

    def drop_columns_constant_values(self, dataframes_dict: dict, ignore_columns: list, drop_columns: bool = True,
                                     print_columns: bool = True,
                                     _reference: Union[bool, str] = False) -> Tuple[Dict, List]:
        """ Constant value features eliminator

        :param dict dataframes_dict: A dictionary that contains Pandas dataframes
                e.g. dataframes_dict={"train": train_dataframe, "test": test_dataframe}
        :param list ignore_columns: It contains the columns that should be ignored e.g. the id and the target.
        :param bool drop_columns: If true, the columns that contain constant values along all the rows will be dropped.
        :param bool print_columns: If true, information about the columns that contain constant values will be printed to the console
        :param Union[bool, str] _reference: The reference dataframe which is used when applying functions to other dataframes. The default value is the first dataframe inside the dataframes_dict dictionary. Usually it is the train dataframe

        :return:
                - dataframes_dict - A dictionary that contains Pandas dataframes after dropping features with constant values e.g. dataframes_dict={ "train": train_dataframe, "test": test_dataframe}
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.
        """

        function_id = "7"

        if not _reference:
            _reference = next(iter(dataframes_dict.keys()))
        dataframes_dict[_reference] = drop_const_columns(dataframes_dict[_reference],
                                                         drop_columns,
                                                         print_columns)

        dataframes_dict = unify_dataframes(dataframes_dict, _reference, ignore_columns)

        self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return dataframes_dict, self.columns_set

    def update_data_summary(self, dataframes_dict: dict) -> list:
        """Data type updater

        This function update the list of the features in the columns types dictionary. This function should be used in
         case of modifying the features in a  dataset manually. For example, dropping some features or after joining two
         datasets.

        :param dict dataframes_dict: A dictionary that contains Pandas dataframes  e.g. dataframes_dict={"train":
                train_dataframe, "test": test_dataframe}
        :return:
                - columns_set - A dictionary that contains the features' names sorted in multiple lists based on the type of the data for each given dataset.
        """

        function_id = "8"

        self.columns_set = detect_columns_types_summary(dataframes_dict, threshold=self.categorical_threshold)
        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return self.columns_set

    def save_post_processed_data(self, dataframe_dict: dict,
                                 path: str,
                                 suffix: str = "",
                                 directory_name: str = "post_processing"):
        """ Data saver

        The function saves data after applying the processing functions.

        :param dict dataframe_dict: A dictionary that contains Pandas dataframes  e.g. dataframes_dict={"train":
                train_dataframe, "test": test_dataframe}
        :param str path: The path to the data.
        :param str suffix: A string that can be added to the filename to get unique names.
        :param str directory_name: The subdirectory where the post-processing data will be saved
        :return:
        """

        function_id = "9"

        path = os.path.join(path, directory_name)
        print(f"Check if the directory {path} exists")

        if not os.path.exists(path):
            os.makedirs(path)

        print(f"{suffix} will be added to the end of the filename")

        filename_list_str = ""

        for key_i, dataframe in dataframe_dict.items():
            filename = key_i + "_" + suffix + ".csv"
            path_i = os.path.join(path, filename)

            if filename_list_str == "":
                filename_list_str = filename
            else:
                filename_list_str = filename_list_str + ", " + filename

            print(f"saving {key_i} dataset to {path_i} file")
            dataframe.to_csv(path_i, index=False)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")
            FlowInstructions.read_data(specific_directory=path, specific_file_list=f"{filename_list_str}")
        pass

    def outliers_extractor(self, dataframe_dict: dict, key_i: str) -> Tuple[pd.Series, List]:
        """ Outliers extractor

        This functions detect if there is outliers in the given dataset. It uses the Robust Random Cut Forest Algorithm.
        For more information about this method please check the following link\n
        https://klabum.github.io/rrcf/random-cut-tree.html

        :param dict dataframe_dict: A dictionary that contains Pandas dataframes  e.g. dataframes_dict={"train":
                train_dataframe, "test": test_dataframe}
        :param str key_i: It points to the dataset which the user wants to explore e.g. "train".
        :return:
                - pd.series avg_codisp: The averege score of each data point calculated from all the constructed trees
                - list outliers_index: bool type list. If the value is true, then the data point is an outlier.
        """

        function_id = "10"
        data_points = dataframe_dict[key_i][self.columns_set[key_i]["continuous"] +
                                            self.columns_set[key_i]["categorical_integer"]]
        data_points = data_points.fillna(data_points.mean()).to_numpy()
        average_score, is_outlier = outliers_detector(data_points)

        try:
            self.guidance(self.flow_steps[function_id])
        except Exception as e:
            print(f"It seems that the next step is not defined. Error{e}")

        return average_score, is_outlier
