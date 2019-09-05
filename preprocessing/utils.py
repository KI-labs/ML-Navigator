import os

import pandas as pd


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
