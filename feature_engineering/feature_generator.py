import logging
import os

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder

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


def valid_features_detector(dataframe: dict, categorical_features: list,
                            class_number_range: list,
                            ignore_columns: list) -> list:
    """ Feature validator

    The functions checks if the one-hot encoding should be applied to the given features.

    :param dict dataframe: A pandas dataframe which contain the dataset
    :param list categorical_features: A list of string that contains the name of the columns or features that contain
            categorical data type.
    :param list class_number_range:
    :param list ignore_columns: list of strings which are the columns names. One-hot encoding will not be applied to
            those columns.
    :return:
            valid_features: A list of the features which one-hot encoding will be applied to.
    :rtype: list
    """

    valid_features = [x for x in categorical_features if x not in ignore_columns]
    valid_features = [x for x in valid_features if len(dataframe[x].value_counts()) in range(class_number_range[0],
                                                                                             class_number_range[1])]
    return valid_features


def one_hot_encoding_sklearn(dataframes_dict: dict, reference: str, categorical_features: list,
                             class_number_range: list,
                             ignore_columns: list) -> dict:
    """ One-hot encoder

    The function applies one-hot encoding to the categorical features using the Scikit Learn framework implementation.

    :param dict dataframes_dict: A dictionary that contains the dataframes before applying one-hot encoding e.g.
            dataframes_dict={ 'train': train_dataframe, 'test': 'test_dataframe'}
    :param str reference: The name of the dataframe that will be considered when validating the type of the data
    :param list categorical_features: A list of string that contains the name of the columns or features that contain
            categorical data type.
    :param list class_number_range: A list that contains two integers which refer ot the range of the minimum and the
            maximum number of the labels/classes/ categories. If a number of the categories of the feature is not in
            that defined range, one-hot encoding will be not applied to that feature.
    :param list ignore_columns: list of strings which are the columns names. One-hot encoding will not be applied to
            those columns.
    :return:
            dataframes_dict_one_hot: A dictionary that contains the dataframes after applying one-hot encoding e.g.
            dataframes_dict_one_hot={ 'train': train_dataframe, 'test': 'test_dataframe'}

    :rtype: dict
    """

    dataframes_dict_one_hot = {}

    considered_features = valid_features_detector(dataframes_dict[reference],
                                                  categorical_features,
                                                  class_number_range,
                                                  ignore_columns)

    logging.debug(f"considered_features = {considered_features}")

    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(
        np.array(dataframes_dict[reference][considered_features]))

    for key_i, dataframe in dataframes_dict.items():
        one_hot_encoded_data = pd.DataFrame(one_hot_encoder.transform(np.array(dataframe[considered_features])))
        columns = [f"col_one_hot_{x}" for x in range(one_hot_encoded_data.shape[1])]

        logging.debug(f"the number of the one-hot encoded data columns is {len(columns)}")

        one_hot_encoded_data.columns = columns

        # Encoded features will be dropped.
        dataframe = dataframe.drop(considered_features, axis=1)

        dataframes_dict_one_hot[key_i] = pd.concat([dataframe, one_hot_encoded_data], axis=1, sort=False)

    return dataframes_dict_one_hot

def target_based_encoding():

    pass