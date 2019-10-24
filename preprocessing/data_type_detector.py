import logging
import os

from IPython.display import display
import pandas as pd

from preprocessing.json_preprocessor import feature_with_json_detector
from preprocessing.data_explorer import outliers_detector

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


class ColumnDataFormat:
    """ Data type detector

    This class contains methods that help to detect the type of the data in each column of the given dataframe. The
    supported data types for now are: date, categorical feature with string values, numeric features with integer
    values (categorical), numeric with continuous values, and nested JSON data format.

    :param: pd.DataFrame dataframe: A pandas dataframe that contains the dataset e.g. train_dataframe.

    :methods: - `find_date_columns` - Date data type finder
              - `number_or_string` - Numeric-string finder
              - `json_detector`- Valid JSON data finder
              - `categorical_or_numeric` - Numeric continuous-discrete finder
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        :param pd.DataFrame dataframe:  A pandas dataframe that contains the dataset e.g. train_dataframe
        """

        self.dataframe = dataframe

    def find_date_columns(self) -> list:
        """ Date data type finder

        This method finds date columns automatically.

        :return:
                date_columns: list of columns that contain date format data
        """

        logger.info("Looking for date columns")

        def look_for_date(column_i: pd.Series):
            dates = {date: pd.to_datetime(date) for date in column_i.unique()}
            return column_i.apply(lambda x: dates[x])

        date_columns = []
        possible_date = list(self.dataframe.select_dtypes(include=["datetime"]).columns)
        if possible_date:
            logger.info("Date columns with native date format was found")
            logger.debug(
                f"there are {len(possible_date)} date column with native format (datetime)"
            )

            date_columns = [x for x in possible_date]

            logger.debug(
                f"the columns that contain native date format are {date_columns}"
            )

        for col in self.dataframe.select_dtypes(include=["object"]).columns:
            try:
                self.dataframe[col] = look_for_date(self.dataframe[col])
                date_columns.append(col)
                logger.info(f"column {col} has date data type")
            except ValueError:
                logger.debug(f"{col} has no date data type")
                pass
        return date_columns

    def number_or_string(self, date_columns: list):
        """ Numeric-string finder

        The function extracts which columns in the pandas dataframe contain numeric values and which have string values.
        It returns three lists of strings

        :param list date_columns: contains the name of the columns that have date format data to exclude those
                columns from the search.

        :return:
                - string_columns - A list contains the column names that contain string type data.
                - numeric_columns - contains the list of columns that have numbers.
                - other_columns - contains the name of columns that have unknown type of data if they exist
        """

        string_columns = []
        numeric_columns = []
        other_columns = []

        columns_to_consider = [
            x for x in self.dataframe.columns if x not in date_columns
        ]

        regex_for_numeric = r"[-+]?[0-9]*\.?[0-9]*"
        for column_i in columns_to_consider:

            if self.dataframe[column_i].dropna().dtype == object:
                string_columns.append(column_i)
                continue

            if (
                    self.dataframe[column_i].dropna().astype(str).str.contains(regex_for_numeric, regex=True).all()
            ):
                numeric_columns.append(column_i)
                continue

            other_columns.append(column_i)

        return string_columns, numeric_columns, other_columns

    def json_detector(self, columns_with_strings: list):
        """ Valid JSON data finder

        This method detects if there is valid nested JSON data inside the columns that have string data. It return Two
        lists of strings

        :param list columns_with_strings: List of the columns that contain string data
        :return:
                - string_columns - A list contains the name of the columns that don't have valid JSON nested data.
                - json_columns - A list contain the name of the columns that have valid nested JSON data.
        """

        json_columns = []
        string_columns = []

        for column_i in columns_with_strings:
            try:
                if feature_with_json_detector(self.dataframe[column_i]):
                    json_columns.append(column_i)
                else:
                    string_columns.append(column_i)
            except:
                print("This column cannot be checked for json data type:", column_i)
                print("it is considered as an object type")
                string_columns.append(column_i)

        return string_columns, json_columns

    def categorical_or_numeric(self, numbers_column_list: list, threshold: float):
        """ Numeric continuous-discrete finder

        The function extracts the name of the columns that contain numeric discrete values and the columns that have
        numeric continuous values from the columns that contain only number data type. The decision is based on the
        unique values in that column. If the number of the unique values is less than the pre-defined threshold,
        the column type will be considered categorical. It returns two lists of strings.

        :param list numbers_column_list: A list of strings which are the names of the columns that contain number value
                type.
        :param int threshold: It is the minimum number of the unique values that under it the column type will be
                considered categorical.

        :return:
                - categorical_columns - the list of the columns' names of the columns that contain numeric discrete data.
                - numeric_columns - The list of the columns' names that contain numeric continuous data.
        """

        categorical_columns = []
        numeric_columns = []

        for column_i in numbers_column_list:

            if len(self.dataframe[column_i].value_counts()) <= threshold:
                categorical_columns.append(column_i)

            else:
                numeric_columns.append(column_i)

        return categorical_columns, numeric_columns


def detect_column_types(dataframe: pd.DataFrame, threshold: int = 50):
    """ Features' types detector

    This function applies the methods defined in the `ColumnDataFormat` class to detect data format in each column.

    :param pd.DataFrame dataframe: A pandas dataframe that contains the dataset e.g. train_dataframe
    :param int threshold: the minimum number of the unique values that under it the column type will be
            considered categorical. The default value here is 50. This becomes very important when applying one-hot
            encoding.
    :return:
            - number_of_columns - An integer which is the total number of features. It is used for the validation purpose.
            - columns_types_list - A list of lists:
            - string_columns - A list of strings which are the columns that contain categorical data type with string labels  e.g. Yes, No, Maybe.
            - categorical_integer - A list of strings which are the columns that contain categorical data type with numeric labels e.g. 0, 1, 2
            - numeric_columns - A list of strings which are the columns that contain columns contains numeric continuous values e.g. float like 0.1, 0.2 or large number of labels of numeric categorical data (larger than the threshold).
            - date_columns - A list of strings which are the columns that contain columns contain date format data. e.g. 2015-01-05
            - other_columns - A list of strings which are the columns that contain columns that has some other types ( Not implemented yet)
    """

    # create an object
    column_detector = ColumnDataFormat(dataframe)

    # find the date columns first and pass it later to other functions
    date_columns = column_detector.find_date_columns()

    # find columns that have strings or numbers first (other columns will be used later when detecting JSON nested
    # columns)
    string_columns, numeric_columns, other_columns = column_detector.number_or_string(
        date_columns
    )

    # for the numeric columns, find out if they have a large number of unique values or not.
    categorical_integer, numeric_columns = column_detector.categorical_or_numeric(
        numeric_columns, threshold
    )

    # for the string columns, find out if there is valid JSON nested data

    string_columns, json_columns = column_detector.json_detector(string_columns)

    columns_types = {
        "categorical_string": string_columns,
        "categorical_integer": categorical_integer,
        "continuous": numeric_columns,
        "date": date_columns,
        "json": json_columns,
        "other": other_columns
    }

    number_of_columns = sum([len(x) for x in columns_types.values()])

    if number_of_columns != dataframe.shape[1]:
        raise ValueError("Number of columns must be equal to the dataframe's columns")

    return columns_types, number_of_columns


def detect_columns_types_summary(dataframes_dict: dict, threshold: int = 50) -> dict:
    """Data type summarizer

    This function summarize the findings after applying the `detect_column_types` function to each given dataset.

    :param dict dataframes_dict: a dictionary of pandas dataframes e.g. {"train": train_dataframe,
            "test": test_dataframe}
    :param int threshold: The maximum number of categories that a categorical feature should have before considering
            it as continuous numeric feature.
    :return:
            columns_types_dict: A dictionary that contains the lists of the columns filtered based on the type of the
            data that they contain.
    """

    columns_types_dict = {}
    for key_i, dataframe in dataframes_dict.items():
        columns_types, _ = detect_column_types(
            dataframe, threshold=threshold
        )

        columns_types_dict[key_i] = columns_types

    print(f"A summary of the data sets")

    data = dict((k, dict((k_, len(v_)) for k_, v_ in v.items())) for k, v in columns_types_dict.items())

    for k, v in data.items():
        v['total amount'] = sum(v.values())

    summary = pd.DataFrame(data)
    summary.index.name = 'column type'
    display(summary)

    print(
        f"\033[1mNOTE: numeric categorical columns that contains more than {threshold} "
        "classes are considered numeric continuous features.\033[0;0m"
    )
    print(
        "\033[1mNOTE: You can modify the threshold value if you want to consider more or less numeric categorical "
        "features as numeric continuous features.\033[0;0m"
    )
    print("Applying Robust Random Cut Forest Algorithm for outliers detection")
    print("Only 'continuous' and 'categorical_integer' are considered for outliers detection")
    for key_i, dataframe in dataframes_dict.items():
        data_points = dataframe[columns_types_dict[key_i]["continuous"] +
                                columns_types_dict[key_i]["categorical_integer"]]

        data_points = data_points.fillna(data_points.mean()).to_numpy()

        _, is_outlier = outliers_detector(data_points)
        if sum(is_outlier) > 0:
            print(
                f"\033[1mNOTE: The dataset {key_i} may have {sum(is_outlier)} outliers.\033[0;0m"
            )

    return columns_types_dict
