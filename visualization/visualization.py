import logging
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact
from matplotlib import colors as m_colors

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


def get_features_set(dataframes_dict: dict):
    features_list = []
    for key_i, dataframe in dataframes_dict.items():
        features_list += list(dataframe.columns)

    features_set = set(features_list)

    return features_set


def histogram(data: pd.Series):
    """ Histogram plotter

    This function plots the histogram of the numeric values of a certain feature when calling the Interactive data
    explorer function in the preprocessing package.

    :param pd.Series data: The values of the given feature/column from the dataset

    """

    _, _, _ = plt.hist(data, 50, facecolor="g", alpha=0.75)
    print(
        "If you applied a normalization function, the x axis will show the normalized values."
    )
    plt.xlabel("Feature's values")
    plt.ylabel("Counts (Frequency)")
    plt.title("Histogram")
    plt.grid(True)
    plt.show()

    pass


def explore_missing_values(dataframes_dict: dict, number_of_features: int):
    """ Missing values explorer

    This function plots the amount of missing values for given amount of features among all datasets.

    :param dict dataframes_dict: A dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}
    :param int number_of_features: The number of the features that should be shown in the plot.

    """

    if number_of_features <= 0:
        logger.error("number of features is invalid")
        raise ValueError("The number of features must be an integer positive and larger than zero")

    maximum_missing_values = 0
    number_of_datasets = len(dataframes_dict.keys())

    logger.debug(f"there are {number_of_datasets} dataframes for exploring the missing values")

    sns.set(style="whitegrid")
    f, ax_set = plt.subplots(number_of_datasets, 1, figsize=(10, 10), sharex=True)

    counter = 0
    g = None

    for key_i, dataframe in dataframes_dict.items():

        missing = dataframe.isnull().sum()
        missing.sort_values(inplace=True, ascending=False)
        missing = missing[:number_of_features]

        logger.debug(f"The number of columns to plot is {number_of_features}")

        g = sns.barplot(x=missing.index, y=missing, palette="deep", ax=ax_set[counter])
        ax_set[counter].axhline(0, color="k", clip_on=False)
        ax_set[counter].set_ylabel(f"{key_i} Count missing values")
        if max(missing) > maximum_missing_values:
            maximum_missing_values = max(missing)

            logger.debug(f"maximum count of missing value is {maximum_missing_values}")

        counter += 1
    # Finalize the plot
    for dataframe_nr in range(len(dataframes_dict.keys())):
        ax_set[dataframe_nr].set(ylim=(0, maximum_missing_values))
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)

    pass


colors = list(dict(m_colors.BASE_COLORS, **m_colors.CSS4_COLORS).keys())


class CompareStatistics:
    """ Statistic explorer

    The class contains methods for exploring the statistics of a given feature.

    :param dict dataframes_dict: a dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}

    """

    def __init__(self, dataframes_dict: dict):
        """

        :param dataframes_dict: A dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}
        """

        self.dataframes_dict = dataframes_dict
        self.features_set = get_features_set(dataframes_dict)

    def check_feature_valid(self, feature: str) -> bool:
        """ Feature's value validator

        The function validate if it is possible to derive the statistical properties of a given feature.

        :param int feature: The index of the column where the feature is.

        :return:
                True if it is possible to calculate the statistical properties of the given feature. Otherwise false.
        """

        for key_i, dataframe in self.dataframes_dict.items():
            try:
                if (dataframe[feature].dropna().dtype == object) or \
                        (dataframe[feature].dropna().dtype == '<M8[ns]') or \
                        (dataframe[feature].dropna().dtype == '>M8[ns]'):
                    print(
                        f"{feature}: You need to encode the feature's values before plotting statistics"
                    )
                    return False
            except Exception as e:
                print(f"This feature can't be found in all dataframes: {e}")
                return False
        return True

    def compare_statistics_function(self, feature_nr: int):
        """Statistic plotter

        This function plots the statistical values of a certain feature among all given datasets in a single graph.

        :param int feature_nr: The index of the column where the feature is.

        """

        feature = list(self.features_set)[feature_nr]
        if CompareStatistics.check_feature_valid(self, feature):

            fig, ax = plt.subplots()
            counter = 0
            ind = 0

            number_of_dataframes = len(self.dataframes_dict.keys())
            width = 0.7 / number_of_dataframes  # the width of the bars

            for key_i, dataframe in self.dataframes_dict.items():
                try:
                    dataframe_statistic = dataframe.describe().loc[
                                          ['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :]
                    if counter == 0:
                        print(f"{feature}: Comparing the statistical properties")
                        ind = np.arange(len(dataframe_statistic[feature]))  # the x locations for the groups
                    _ = ax.bar(ind - (number_of_dataframes - 1) * width / 2 + counter * width,
                               dataframe_statistic[feature], width,
                               color=colors[counter], label=key_i)
                    counter += 1
                except Exception as e:
                    print(f"The Error is: {e}")

            ax.set_xticklabels(('count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'))
            ax.legend()


def compare_statistics(dataframes_dict: dict):
    """ Interactive statistic explorer

    This function is designed to be run in a Jupyter notebook. The user can go through the feature interactively using
    a slider.

    :param dict dataframes_dict: A dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}

    """

    compare_statistics_object = CompareStatistics(dataframes_dict)

    maximum_number_features = len(get_features_set(dataframes_dict))

    interact(compare_statistics_object.compare_statistics_function,
             feature_nr=widgets.IntSlider(min=0, max=maximum_number_features - 1, step=1, value=0))
