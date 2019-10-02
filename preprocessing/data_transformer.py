import logging
import os
import itertools
import random
from typing import Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
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
pd.set_option('display.max_rows', None)


def standard_scale_numeric_features(
        dataframe_dict: dict,
        reference_dataframe_key: str,
        columns_to_normalize: list,
        handle_missing_values: bool = True,
) -> dict:
    """ Feature standardizer

    This function standardizes the datesets passed as pandas dataframes by a dictionary. The reference dataframe will be
    used to calculate the statistical properties (mean and standard deviation) and used to normalize other
    dataframes. After standardizing the dataset, the features in each dataset have 0 mean and 1 standard deviation.

    :param dict dataframe_dict: A dictionary that contains multiple pandas dataframes
    :param str reference_dataframe_key: A string that is used to fit the scaler e.g. "train"
    :param list columns_to_normalize: A list of the columns that should be standardized e.g. [col_1, col_2, ..., col_n]
    :param bool handle_missing_values: boolean if true the missing values will be replaced by the value 0

    :return:
        scaled_dataframe_dict: A dictionary of pandas dataframes where the "columns_to_normalize" are normalized
    :rtype: dict
    """

    scaler = None
    scaled_dataframe_dict = {}

    # For scaling the data, the reference, that will be used to fit the StandardScaler, should be selected.
    reference_dataframe = dataframe_dict[reference_dataframe_key]

    # fitting the StandardScaler
    try:
        scaler = preprocessing.StandardScaler().fit(
            reference_dataframe[columns_to_normalize]
        )
    except Exception as e:
        logger.error(f"Error is: {e}")

    # Scaling data
    for key_i, dataframe in dataframe_dict.items():
        scaled = scaler.transform(dataframe[columns_to_normalize])

        # After scaling, the mean values will be 0. Therefore, the missing values will be replaced by the mean value
        if handle_missing_values:
            logger.info("Missing values will be handled")
            scaled[np.isnan(scaled)] = 0

        dataframe[columns_to_normalize] = scaled
        scaled_dataframe_dict[key_i] = dataframe

    logger.info("Scaling data is finished")
    return scaled_dataframe_dict


def encoding_categorical_feature(dataset_dict: dict, feature_name: str,
                                 print_results: Union[bool, int] = True,
                                 print_counter: int = 0) -> dict:
    """ Single categorical feature string encoder

    This function encodes categorical features. It is possible to use train data alone or all train data, validation
    data and test data. If all datesets are provided (i.e. train, valid and test), they will be concatenated first
    and then encoded.

    :param int print_counter: if print_results is int, print counter control printing data to the conosle based on the print_results value.
    :param Union[bool, int] print_results: If False, no data is printed to the console. If True, all data is printed to the console. If an integer n, only the data for n features is printed to the console.
    :param str feature_name: The name of the feature/column that its values should be encoded.
    :param dict dataset_dict: a dictionary of pandas series (i.e one column) that must contain the train data and
                                optionally contains valid data and test data

    :return:
            dataset_dict_encoded: a dictionary of pandas series (i.e one column) after encoding.
    """

    # Replacing the missing values with a special hash value to avoid having the same class of missing values and
    # non-missing values.
    hash_missing_value = hex(random.getrandbits(128))
    logger.debug(f"The hash for the missing values is {hash_missing_value}")

    # Concatenate datasets if needed

    if isinstance(print_results, bool) and print_results:
        print(f"there are {len(dataset_dict)} datasets provided")
    elif isinstance(print_results, int):
        if print_counter < print_results:
            print(f"there are {len(dataset_dict)} datasets provided")

    # the check here is not a good idea because it check each column alone which is not efficient
    valid_dataset_list = []
    valid_dataset_keys = []

    for key_i, dataseries in dataset_dict.items():
        if dataseries.shape[0] > 0:
            valid_dataset_list.append(dataseries)  # get the dataframes
            valid_dataset_keys.append(key_i)  # get the keys

    if len(valid_dataset_list) > 1:
        x_original = pd.concat(valid_dataset_list, axis=0)
    elif len(valid_dataset_list) == 1:
        x_original = valid_dataset_list[0]
    else:
        raise ValueError("No valid dataset was provided")

    # define the encoder
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(x_original.fillna(hash_missing_value))

    dataset_dict_encoded = {}

    for dataset_key in valid_dataset_keys:
        dataset_dict_encoded[dataset_key] = label_encoder.transform(
            dataset_dict[dataset_key].fillna(hash_missing_value)
        )

        labels_nr = len(list(label_encoder.classes_))
        if isinstance(print_results, bool) and print_results:
            print(f"encoding the feature in the dataset {dataset_key}")
            print(f"the number of classes in {feature_name} feature is: {labels_nr}")

        elif isinstance(print_results, int):
            if print_counter < print_results:
                print(f"encoding the feature in the dataset {dataset_key}")
                print(f"the number of classes in {feature_name} feature is: {labels_nr}")

    logger.info(f"Encoding categorical feature {feature_name} process is finished!")
    return dataset_dict_encoded


def encode_categorical_features(dataframe_dict: dict, columns_list: list,
                                print_results: Union[bool, int] = True) -> dict:
    """ Categorical features string encoder

    This function applies the `encoding_categorical_feature` function to each feature in the `columns_list`.

    :param Union[bool, int] print_results: If False, no data is printed to the console. If True, all data is printed to
        the console. If an integer n, only the data for n features is printed to the console.
    :param dict dataframe_dict: a dictionary of Pandas dataframes.
    :param list columns_list: The list of the names of the columns/features that their values should be encoded.

    :return:
            dataframe_dict: a dictionary of Pandas dataframes after encoding.
    """

    print_counter = 0
    for feature_name in columns_list:
        dataset_dict = {}
        try:
            for key_i, dataframe in dataframe_dict.items():
                dataset_dict[key_i] = dataframe[feature_name].astype(str)

            dataset_dict_encoded = encoding_categorical_feature(dataset_dict, feature_name,
                                                                print_results,
                                                                print_counter)
            for key_i, dataseries in dataset_dict_encoded.items():
                dataframe_dict[key_i][feature_name] = dataseries

            logger.info("Encoding all categorical features process is finished!")

        except Exception as e:
            logger.error(f"The Error: {e}")

        print_counter += 1

    if isinstance(print_results, int) and print_results > 0:
        print("the value is considered integer")

        dfs = [v[[c for c in v.columns if c in columns_list]].nunique().to_frame(name=k)
               for k, v in dataframe_dict.items()]
        result = pd.concat(dfs, axis=1, sort=False)

        dfs = [v[[c for c in v.columns if c in columns_list]].apply(lambda x: list(set(x)), axis=0).to_frame(name=k)
               for k, v in dataframe_dict.items()]

        result_all = pd.concat(dfs, axis=1, sort=False)
        result_all = result_all \
            .apply(lambda x: len(set(itertools.chain(*filter(lambda y: type(y) == list, x)))), axis=1) \
            .to_frame(name='all data sets')

        result = pd.concat([result, result_all], axis=1, sort=False)
        result.index.name = 'string columns'

        print(f"Number of unique elements per string column")

        if print_results > 1:
            pd.set_option('display.max_rows', print_results)
        display(result)
        pd.set_option('display.max_rows', None)

    return dataframe_dict
