import logging
import os
from typing import Tuple, List

import ipywidgets as widgets
import numpy as np
import pandas as pd
import rrcf
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


class OutliersDetection:

    def __init__(self, data_array: np.array, number_trees: int = 100, tree_size: int = 128):
        """ Parameters initiator

        :param np.array data_array: A numpy array which has the size of n x m. The missing values are replaced by the
                                    means and the elements are only numeric
        :param int number_trees: The number of the trees that will be randomly generated to create the forest
        :param int tree_size: The number of the elements that a tree should contain.
        """

        self.data_array = data_array
        self.number_trees = number_trees
        self.tree_size = tree_size
        self.number_of_rows = self.data_array.shape[0]
        self.number_of_feature = self.data_array.shape[1]

    def construct_force(self) -> List:
        """ Forest creator

        This function creates a list of trees which are constructed randomly.

        :return:
                - list forest: A list of the trees that are created randomly
        """

        forest = []

        sample_size_range = (self.number_of_rows // self.tree_size, self.tree_size)

        if sample_size_range[0] == 0:
            raise ValueError("Please check the tree_size. It seems that the number of the "
                             "samples (rows) is less than the size of the tree")

        while len(forest) < self.number_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(self.number_of_rows, size=sample_size_range,
                                   replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(self.data_array[ix], index_labels=ix)
                     for ix in ixs]
            forest.extend(trees)
        return forest

    def codisp_average(self, forest: list) -> pd.Series:
        """ Average score

        It calculates the score for each data point in all the trees and then calculates the average value of the score.

        :param list forest: A list of the trees that are created randomly.
        :return:
                - pd.series avg_codisp: The averege score of each data point calculated from all the constructed trees
        """

        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(self.number_of_rows))
        index = np.zeros(self.number_of_rows)
        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        return avg_codisp


def outliers_detector(data_array: np.array,
                      number_trees: int = 100,
                      tree_size: int = 128,
                      outliers_threshold: float = 0.99) -> Tuple[pd.Series, List]:
    """ RRCF outliers detector

    This functions detect if there is outliers in the given dataset. It uses the Robust Random Cut Forest Algorithm.
    For more information about this method please check the following link\n
    https://klabum.github.io/rrcf/random-cut-tree.html

    :param np.array data_array: A numpy array which has the size of n x m. The missing values are replaced by the means
                                and the elements are only numeric
    :param int number_trees: The number of the trees that will be randomly generated to create the forest
    :param int tree_size: The number of the elements that a tree should contain.
    :param float outliers_threshold: The score' threshold which above it, the data point is considered as an outlier.
    :return:
            - pd.series avg_codisp: The averege score of each data point calculated from all the constructed trees
            - list outliers_index: bool type list. If the value is true, then the data point is an outlier.
    """

    avg_codisp = []
    outliers_index = []

    try:
        # Construct forest
        detect_outliers = OutliersDetection(data_array, number_trees, tree_size)
        forest = detect_outliers.construct_force()
        avg_codisp = detect_outliers.codisp_average(forest)

        outliers_index = avg_codisp > avg_codisp.quantile(outliers_threshold)
    except Exception as e:
        print("It is not possible to figure out outliers")
        print("Error: ", e)
    return avg_codisp, outliers_index
