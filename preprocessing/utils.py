import os
import pandas as pd
import numpy as np
from IPython.display import display


def read_data(path: str, files_list: list, rows_amount: int = 0) -> dict:
    """ CSV file reader

    This function reads CSV files that their names are listed inside the `file_list`

    :param str path: It points to the directory where the data is stored.
    :param list files_list: A list of strings which are the names of the files.
    :param int rows_amount: The number of rows that should be read from the CSV file. If 0, all rows will be read.
    :return:
            dataframes_dictionary: A dictionary that contains Pandas dataframes. The keys are the name of the files
            without the csv extension and the values are the associated dataframes.
    :raise:
            - ValueError - In case of rows_amount has invalid value

    :example:

    >>> path = "./data"
    >>> files_list = ["train.csv", "test.csv"]
    >>> dataframes_dictionary = {"train": train_dataframe, "test": test_dataframe}
    """

    dataframes_dictionary = {}
    for file_i in files_list:
        if rows_amount == 0:
            dataframes_dictionary[file_i.split(".")[0]] = pd.read_csv(os.path.join(path, file_i))
        elif rows_amount > 0:
            try:
                dataframes_dictionary[file_i.split(".")[0]] = pd.read_csv(os.path.join(path, file_i), nrows=rows_amount)
            except Exception as e:
                print(f"Error loading the data using positive rows_amount value. Error: {e}")
        else:
            raise ValueError("rows_amount has invalid value. rows_amount should be 0 or positive number")
    return dataframes_dictionary


def check_if_target_columns_are_imbalanced(dataframes_dict: dict,
                                           possible_problems: dict,
                                           kl_div_threshold: float = 0.05) -> dict:
    """ Detect class imbalance for categorical target variables

    Check if target categorical columns are imbalanced (i.e the different target values do not
    appear with the same frequency).

    :param dict dataframes_dict: dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}
    :param dict possible_problems: dictionary of target candidates associated with the problem type,
            e.g. {"isFraud": "classification", "total amount": "regression"}
    :param float kl_div_threshold:
            threshold above which the Kullback-Leibler Divergence between the ideal and observed distribution is too big.
            The ideal distribution has equal probabilities for all observed values of the target variable.
    :return:
            target_summary dict: summary of value counts for the different categorical target variables.
    """

    def kl_divergence(p, q):
        """ Kullback-Leibler Divergence

        Note that in our case p and q have always non-zero values.
        """
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    target_summary = {}
    # at the moment we take into account only classification problems
    try:
        possible_clf_problems = dict((k, v) for k, v in possible_problems.items() if 'classification' == v)
    except Exception as e:
        print("Not possible to check if the target is imbalanced")
        print("Error is:\n", e)
        return target_summary

    # for every target column calculate the counts of distict values
    for target in possible_clf_problems:
        value_counts = [df[target].value_counts().to_frame(name=f"{target} {key_i}")
                        for key_i, df in dataframes_dict.items()
                        if target in df.columns]

        value_counts = pd.concat(value_counts, axis=1).fillna(0)
        value_counts['total count'] = value_counts.sum(axis=1)  # sum value counts from all data frames
        value_counts.index.name = f"{target} values"  # index equal to the distinct target values
        value_counts = value_counts[['total count']]
        value_counts["frequency (%)"] = 100 * value_counts['total count'] / value_counts['total count'].sum()
        value_counts["frequency (%)"] = value_counts["frequency (%)"]

        target_summary[target] = {'content': value_counts}

        # calculate the Kullback-Leibler Divergence for categorical problems
    for k, v in target_summary.items():
        df = v["content"]
        p_ideal = np.ones(df.shape[0]) / df.shape[0]  # ideal case with evenly distributed values
        p_real = df["total count"].values / df["total count"].sum()
        v["kl div"] = kl_divergence(p_ideal, p_real)

    # Check if the values of the target variables are ballanced.
    for k, v in target_summary.items():

        kl_div = v['kl div']

        if kl_div < kl_div_threshold:
            print("\nThe Kullback-Leibler Divergence between probability mass function (pmf) derived from `{0:s}` "
                  "and and an uniformly distributed pmf "
                  " \t = {1:.3f} \nIt is below the threshold {2:.3f}"
                  .format(k, kl_div, kl_div_threshold))
        else:
            print("\nThe Kullback-Leibler Divergence between probability mass function (pmf) derived from `{0:s}` "
                  "and and an uniformly distributed pmf "
                  "\t = {1:.3f} \nIt is above the threshold \t {2:.3f}. \n"
                  "Imballanced target variable!".format(k, kl_div, kl_div_threshold))

        display(v["content"].round(2))

    return target_summary
