import logging
import os

import ipywidgets as widgets
import numpy as np
import pandas as pd
from ipywidgets import interact

from visualization.visualization import histogram

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


def print_repeated_values(series_data: pd.Series):
    """ Repeated values displayer

     This function prints out into the console the results of value_counts. It shows count_values.head() or tail()

    :param pd.Series series_data: It the values of one of the features that are in the given dataset.

    """

    try:
        max_space = max([len(str(x)) for x in series_data.index])

        logging.debug(f"The maximum value's length in the given series is {max_space}")

        for value_nr in range(len(series_data)):
            index_i = series_data.index[value_nr]
            print(
                f"the value {index_i} {' ' * (3 + max_space - len(str(index_i)))} is found "
                f"{'' * (1 + max_space - len(str(series_data[index_i])))}"
                f" {series_data[index_i]} times"
            )
    except Exception as e:
        logger.error(f"Error: {e}")


class ExploreData:
    """ Data explorer

    This class have the `data explore` method which can be used to explore the data in each column in the dataset.

    :param pd.DataFrame dataframe: A pandas dataframe that contains the dataset e.g. train_dataframe

    """

    def __init__(self, dataframe: pd.DataFrame):
        """

        :param pd.DataFrame dataframe: A pandas dataframe that contains the dataset e.g. train_dataframe
        """

        self.dataframe = dataframe

    def data_explore(self, column_i: str):
        """ Feature explorer

        This method displays a summary about the given feature including missing values, most and least repeated
        values. Besides that, it shows the histogram of the numeric data.

        :param str column_i: The name of the feature that the user is interested in exploring.

        """

        column_name = self.dataframe.columns[column_i]
        logger.info(f"Exploring the values of the feature {column_name}")

        print(f"\033[1mExploring column: {column_name}\033[0;0m")

        if self.dataframe[column_name].dtype == object:
            logger.debug(f"The type of the data of the feature {column_name} is object")

            # To take the missing value into account, I replace the missing values with "missing values"
            count_values = self.dataframe[column_name].fillna("missing values").value_counts(dropna=False)

            print("If there are missing values, they will be replaced by the term: missing values")

        elif self.dataframe[column_name].dtype == float:
            logger.debug(f"The type of the data of the feature {column_name} is float")

            # To take the missing value into account, I replace the missing values with the mean values
            count_values = self.dataframe[column_name].fillna(np.nanmean(self.dataframe[column_name])).value_counts(
                dropna=False)

            print(f"If there are missing values," +
                  f" they will be replaced by the mean: {np.nanmean(self.dataframe[column_name])}")
        else:
            logger.debug(f"The type of the data of the feature {column_name} is something else than object or "
                         "float-most probably categorical with integer values")

            # To take the missing value into account, I replace the missing values with the value -9999
            count_values = self.dataframe[column_name].fillna(-9999).value_counts(dropna=False)
            print("If there are missing values, they will be replaced by the term: -9999")

        # showing information to the user
        print("The 5 most repeated values:")
        print_repeated_values(count_values.head())
        print("The 5 least repeated values")
        print_repeated_values(count_values.tail())
        print("The number of unique values is -->", len(count_values))

        # In order to show the histogram, the values should be numeric
        if self.dataframe[column_name].dropna().dtype == object:
            logger.debug(f"It is not possible to show the histogram of the feature {column_name}")
            print(
                "You need to encode the feature's values before showing the histogram"
            )
        else:
            logger.debug(f"The feature {column_name} has bool type values")

            # For showing the histogram of the bool values, the True, False values should be replaced by 1, 0 values.
            if self.dataframe[column_name].dropna().dtype == bool:
                tmp_dataframe = self.dataframe[column_name].astype("int")
                print("For visualization True is replaced by 1 and False by 0")
            else:
                tmp_dataframe = self.dataframe[column_name]

            histogram(tmp_dataframe)
        pass


def explore_data(dataframe):
    """ Interactive data explorer

    This function should be run in a Jupyter notebook. The user can go through the feature interactively using a slider.

    :param pd.DataFrame dataframe: A pandas dataframe that contains the dataset e.g. train_dataframe.

    """

    explore_object = ExploreData(dataframe)

    logger.info("Starting the interactive data exploring")
    logger.debug(f"The number of columns is {dataframe.shape[1] - 1}")

    interact(
        explore_object.data_explore,
        column_i=widgets.IntSlider(min=0, max=dataframe.shape[1] - 1, step=1, value=0),
    )
