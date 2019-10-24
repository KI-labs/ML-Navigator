import json
import logging
import os
from typing import Union

import pandas as pd
from pandas.io.json import json_normalize

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


def extract_json_from_list_dict(row_i: Union[list, dict], object_nr: int) -> dict:
    """ Valid JSON object extractor from list or dict

    The function extracts the valid JSON data directly from the list or the dict.

    :param Union[list, dict] row_i: The content of the row which has the index i in a Pandas series
    :param int object_nr: If there are multiple valid JSON objects detected, the object which has object_nr will be
            returned

    :return: valid_json_object that has the index object_nr
    :rtype: dict
    """

    if isinstance(row_i, list) and len(row_i) > object_nr:
        try:
            return json.loads(json.dumps(row_i[object_nr]))
        except Exception as e:
            logger.debug(f"Not valid JSON: {e}")
    elif isinstance(row_i, dict) and object_nr == 0:
        try:
            return json.loads(json.dumps(row_i))
        except Exception as e:
            logger.debug(f"Not valid JSON: {e}")
    return {}


def extract_json_objects(raw_string_data: str, start_json_object: list,
                         end_json_object: list, object_nr: int):
    """Valid JSON object extractor

    The function extracts valid JSON objects from texts. Those objects could not be extracted using the json.load

    :param str raw_string_data: The string object that could contain valid JSON objects
    :param list start_json_object: List of integers that point to the char "{" which can be the start of the JSON
            objects.
    :param list end_json_object: List of integers that point to the char "}" which can be the end of the JSON objects.
    :param int object_nr: points to the index of the object that should be extracted if there are multiple valid JSON
            objects.

    :return: valid_json_object that has the index object_nr
    :rtype: dict
    """

    # initiate variables
    counter = 0
    valid_json_objects = []
    current_position = 0

    for start_i in start_json_object:
        if start_i >= current_position:
            for end_i in end_json_object:
                try:
                    valid_json_object = json.loads(raw_string_data[start_i:end_i + 1])
                    valid_json_objects.append(valid_json_object)
                    current_position = end_i
                    counter += 1
                    break
                except Exception as e:
                    logger.debug(f"Not valid JSON: {e}")
                    # not a valid JSON object
                    continue
        if counter == object_nr + 1:
            return [valid_json_objects[object_nr]]
    return {}


def normalize_feature(string_data: str, object_nr: int):
    """JSON data searcher

    This function searches for possible valid JSON data inside the given text. It identifies the possible JSON objects
    by defining their edges using "{" and "}". It passes each defined object to the "extract_json_objects" function
    to extract the valid JSON objects. The returned valid objects will be normalized and returned as a Pandas dataframe.

    :param str string_data: The string that could contain valid JSON objects.
    :param int object_nr: If there are multiple valid JSON objects detected, the object which has object_nr will be
            returned

    :return: Pandas dataframe which has "n" number of columns and one row.
    :rtype: pandas.DataFrame
    """

    json_valid_object = {}
    try:
        # This will raise an error if the string_data is nan
        if len(string_data) == 0:
            string_data = {}

        # Trying to normalize it directly if the input is not a string type
        json_valid_object = extract_json_from_list_dict(string_data, object_nr)

        # if the regular method fails to extract JSON, try the second
        if json_valid_object == {}:
            # Search for the start positions of the possible valid JSON objects
            start_json = [i for i, x in enumerate(string_data) if str(x) == "{"]

            # Search for the end positions of the possible valid JSON objects
            end_json = [i for i, x in enumerate(string_data) if str(x) == "}"]

            # Pass the string together with the start and end positions to the JSON data extractor
            json_valid_object = extract_json_objects(string_data, start_json, end_json, object_nr)

    except Exception as e:
        logger.debug(string_data)
        logger.debug(e)

    try:
        return json_normalize(json_valid_object)
    except Exception as e:
        logger.debug(f"not valid object to normalize: {json_valid_object}")
        logger.debug(f"the Error is {e}")
        return pd.DataFrame()


def apply_normalize_feature(dataseries: pd.Series, keys_amount: int):
    """JSON-dataframe converter

    :param pd.Series dataseries: the feature that contains possible JSON objects
    :param int keys_amount: The possible numbers of the keys or parent JSON object that a row may contain.

    :return: a list of dataframes. Each element of the this list represents the normalized JSON object in each row of
            the dataseries
    :rtype: list
    """

    # Bring together the JSON objects from each row.
    dataframe_list = []

    for key_i in range(keys_amount):
        # Bring together the JSON objects from one row
        df_list = []

        for row_i in range(dataseries.size):
            string_data = str(dataseries[row_i]). \
                replace("'", '"'). \
                replace("True", '"True"'). \
                replace("False", '"False"'). \
                replace("None", '"None"')
            normalized_object = normalize_feature(string_data, key_i)

            if normalized_object.shape[0] > 0:
                df_list.append(normalized_object)
            else:
                logger.debug("no object was found")

        df = pd.concat(df_list, axis=0, sort=False)
        logger.debug(f"the size of the dataframe from the object {key_i} is {df.shape}")

        # Change the name of the column by adding the key order
        columns = [x + f"_{key_i}" for x in df.columns]
        df.columns = columns
        logger.debug(columns)

        dataframe_list.append(df)
    return dataframe_list


def column_validation(dataframe: pd.DataFrame, parent_columns: list, feature: str):
    """Column name validator

    The function ensures that the dataframe doesn't have two features that have the same name. It changes the name of
    the column after normalizing the JSON object based on the name of the parent feature

    :param pd.DataFrame dataframe: the normalized JSON objects that were found in the given feature.
    :param list parent_columns: A list of the name of the features or columns of main dataset
    :param str feature: The name of the feature that contains the JSON objects

    :return: Pandas dataframe with valid names for the columns
    :rtype: pd.DataFrame
    """

    new_columns = []
    for column_i in dataframe.columns:

        # Give the name of the parent feature to the new extracted column
        column_i = feature + "_" + column_i

        # If the column already exists, add a suffix to the name of the column
        while column_i in parent_columns:
            column_i = column_i + "_1"

        new_columns.append(column_i)

    # Give the dataframe the valid names of the columns
    dataframe.columns = new_columns
    return dataframe


def combine_new_data_to_original(dataframe: pd.DataFrame, dataframe_list: list, feature: str):
    """Dataframes binder

    The function concatenates the original dataframe and the new created dataframe together.

    :param pd.DataFrame dataframe: the original dataframe / dataset
    :param list dataframe_list: list of the dataframes that are created from normalizing the JSON objects in each row
            of the given feature
    :param str feature: The name of the feature that contains JSON objects

    :return: Pandas dataframe that contains both the original and the new created datasets. The original feature will
            be deleted
    :rtype: pd.DataFrame
    """

    # First bring together the dataframes that contain the normalized objects from each row
    char_dataframe = pd.concat(dataframe_list, axis=1, sort=False)

    # Be sure that the index of the new created dataframe is correct and unique
    char_dataframe.index = range(char_dataframe.shape[0])

    # Be sure that the new dataframe has valid columns name (avoid two features have the same name)
    char_dataframe = column_validation(char_dataframe, dataframe.columns, feature)

    logger.debug(f"The shape of dataframe after normalizing the JSON data is: {char_dataframe.shape}")

    # Bring the old and the new datasets together
    joined_dataframe = pd.concat([dataframe, char_dataframe], axis=1, sort=False)

    # Dropping the original feature column after normalizing it.
    joined_dataframe = joined_dataframe.drop(feature, axis=1)

    return joined_dataframe


def feature_with_json_detector(dataseries: pd.Series):
    """ JSON detector

    This function detect if there is features in the dataset that could have valid JSON objects.

    :param pd.Series dataseries: The feature's values that should be tested for possible valid JSON objects

    :return: True if there is JSON objects candidates and False if not
    :rtype: bool
    """

    # Check which row contains start and end JSON syntax in case of string values inside the rows
    start_json = dataseries.fillna('').str.contains('{').fillna("")
    end_json = dataseries.fillna('').str.contains('}').fillna("")
    try:
        if start_json[start_json].size and end_json[end_json].size:
            for row_i in range(start_json.size):
                if start_json[row_i] and end_json[row_i]:
                    raw_string_data = dataseries[row_i]

                    # if the JSON is converted to a dataframe successfully, it is valid JSON
                    if normalize_feature(raw_string_data, 0).shape[0]:
                        return True
    except:
        print("start_json:\n ", start_json)
        print("End_json:\n", end_json)

    # Some of the rows can be valid lists that contain valid JSON format
    if start_json[start_json.isnull()].size and end_json[end_json.isnull()].size:
        for row_i in dataseries[start_json.isnull()]:
            for item_i in range(len(row_i)):
                if extract_json_from_list_dict(row_i, item_i) != {}:
                    return True

    return False


def combine_columns(dataframes_dict, feature):
    """
    For avoiding different numbers of generated columns for different datasets, I combine them in one large dataframe
    :param dataframes_dict:
    :param feature:
    :return:
    """
    dataframes_list = []
    combined_dataframe = pd.DataFrame()
    # I gave a key to easy recognize each dataset later.
    key_name = "key_dataframe"

    for key_i, dataframe in dataframes_dict.items():
        dataframe[key_name] = key_i
        dataframes_list.append(dataframe[[feature, key_name]])
        combined_dataframe = pd.concat(dataframes_list, axis=0, sort=False)
        combined_dataframe.index = range(combined_dataframe.shape[0])

    return combined_dataframe


def flat_json(dataframes_dict, json_columns, keys_amount=10):
    """
    :param dataframes_dict
    :param json_columns:
    :param keys_amount
    :return:
    """

    # Flatten JSON data in a certain feature
    for column_i in json_columns:

        dataframe_combined = combine_columns(dataframes_dict, column_i)

        dataseries = dataframe_combined[column_i]
        print(f"Flatting column: {column_i}")

        dataframe_list = apply_normalize_feature(dataseries, keys_amount)
        dataframe_combined = combine_new_data_to_original(dataframe_combined, dataframe_list, column_i)

        # Extract each data set and delete the added key and the original feature after it was flatten.
        for key_i in dataframes_dict.keys():
            extracted_dataframe = dataframe_combined[dataframe_combined["key_dataframe"] == key_i]

            # index should be reset otherwise there is a problem with the pd.concat
            extracted_dataframe.index = range(extracted_dataframe.shape[0])

            dataframes_dict[key_i] = pd.concat([dataframes_dict[key_i], extracted_dataframe], axis=1, sort=False)

            dataframes_dict[key_i] = dataframes_dict[key_i].drop([column_i, "key_dataframe"], axis=1)

    return dataframes_dict
