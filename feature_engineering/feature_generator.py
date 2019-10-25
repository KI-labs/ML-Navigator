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


def decode_features_with_appearance_frequency(dataframe_dict: dict, reference: str, categorical_features: list) -> dict:
    """ Frequency-based encoder

    :param dict dataframe_dict: A dictionary that contains the dataframes before applying the encoding e.g.
            dataframes_dict={ 'train': train_dataframe, 'test': 'test_dataframe'}
    :param str reference: The name of the dataframe that will be considered when validating the type of the data
    :param list categorical_features: A list of string that contains the name of the columns or features that contain
            categorical data type.
    :return:
            dataframe_dict: A dictionary that contains the dataframes after applying feature encoding
    :rtype: dict

    """

    _reference_dataframe = dataframe_dict[reference]

    for column_i in categorical_features:
        print(f"decoding the column {column_i} based on the value frequency")
        try:
            encoding = _reference_dataframe.groupby([column_i]).size()

            encoding /= _reference_dataframe.shape[0]

            for key_i, dataframe in dataframe_dict.items():
                dataframe[column_i + '_frequency_encoding'] = dataframe[column_i].map(encoding)
                dataframe_dict[key_i] = dataframe

        except Exception as e:
            print(f"error while decoding {column_i} based on value appearance frequency")
            print("Error: ", e)

    return dataframe_dict


def valid_features_detector(dataframe: pd.DataFrame, categorical_features: list,
                            class_number_range: list) -> list:
    """ Feature validator

    The functions checks if the one-hot encoding method should be applied to the given features.

    :param pd.DataFrame dataframe: A pandas dataframe which contain the dataset
    :param list categorical_features: A list of string that contains the name of the columns or features that contain
            categorical data type.
    :param list class_number_range: It is a list of two elements which define the minimum the and maximum number of
                        the classes (unique value) that a feature should contain in order to apply the one-hot encoding
                         to this feature.
    :return:
            valid_features: A list of the features which the encoding will be applied to.
    :rtype: list
    """

    valid_features = [x for x in categorical_features if len(dataframe[x].value_counts()) in range(
        class_number_range[0], class_number_range[1])]
    return valid_features


def encoding_features(encoding_type: str,
                      dataframes_dict: dict,
                      reference: str,
                      categorical_features: list,
                      ignore_columns: list,
                      class_number_range: list = None,
                      target_name: str = None,
                      drop_encoded_features: bool = True) -> dict:
    """ One-hot encoder

    The function applies one-hot encoding to the categorical features using the Scikit Learn framework implementation.

    :param str encoding_type: The type of the encoding method that will be applied. For example: one-hot, target \n
            For more information please check the following reference:\n
            https://contrib.scikit-learn.org/categorical-encoding/index.html
    :param dict dataframes_dict: A dictionary that contains the dataframes before applying the encoding e.g.
            dataframes_dict={ 'train': train_dataframe, 'test': 'test_dataframe'}
    :param str reference: The name of the dataframe that will be considered when validating the type of the data
    :param list categorical_features: A list of string that contains the name of the columns or features that contain
            categorical data type.
    :param list class_number_range: A list that contains two integers which refer ot the range of the minimum and the
            maximum number of the labels/classes/ categories. If a number of the categories of the feature is not in
            that defined range, one-hot encoding will be not applied to that feature.
            It can be ignored if the encoding type is not one-hot.
    :param list ignore_columns: list of strings which are the columns names. The encoding will not be applied to
            those columns.
    :param str target_name: The name of the column that contains the labels that should be predicted by the model.
                            If the encoding method doesn't require that target, it can be ignored.
    :param bool drop_encoded_features: If True, the encoded features will be dropped from the dataset

    :return:
            dataframes_dict_encoded: A dictionary that contains the dataframes after applying feature encoding e.g.
            dataframes_dict_encoded={ 'train': train_dataframe, 'test': 'test_dataframe'}

    :rtype: dict
    """

    dataframes_dict_encoded = {}

    if class_number_range is None:
        class_number_range = [0, 50]

    considered_features = [x for x in categorical_features if x not in ignore_columns]

    encoder = None

    if encoding_type == "one-hot":
        considered_features = valid_features_detector(dataframes_dict[reference],
                                                      categorical_features,
                                                      class_number_range)

        logging.debug(f"considered_features = {considered_features}")

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(
            np.array(dataframes_dict[reference][considered_features]))

    if encoding_type == "target":
        if target_name is None:
            raise ValueError("Please define the target_name. It is the label that should be predicted by the model")

        encoder = ce.target_encoder.TargetEncoder(cols=considered_features).fit(dataframes_dict[reference][considered_features],
                                                                                dataframes_dict[reference][target_name])

    if encoding_type == "frequency":
        dataframes_dict_encoded = decode_features_with_appearance_frequency(dataframes_dict,
                                                                            reference,
                                                                            considered_features)

    else:

        for key_i, dataframe in dataframes_dict.items():
            encoded_data = pd.DataFrame()

            if encoding_type == "one-hot":
                encoded_data = pd.DataFrame(encoder.transform(np.array(dataframe[considered_features])))
            elif encoding_type == "target":
                encoded_data = encoder.transform(dataframe[considered_features])

            columns = [f"col_{encoding_type}_encoding_{x}" for x in range(encoded_data.shape[1])]

            logging.debug(f"the number of the {encoding_type} encoded data columns is {len(columns)}")

            encoded_data.columns = columns

            # Encoded features will be dropped.
            try:
                if drop_encoded_features:
                    dataframe = dataframe.drop(considered_features, axis=1)
            except Exception as e:
                print(f"The feature/s {considered_features} cannot be dropped. The error is:\n {e}")

            dataframes_dict_encoded[key_i] = pd.concat([dataframe, encoded_data], axis=1, sort=False)

    return dataframes_dict_encoded
