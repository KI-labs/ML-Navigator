import logging
import os

import pandas as pd

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


def drop_corr_columns(dataframe: pd.DataFrame, drop_columns: bool = True,
                      print_columns: bool = True, threshold: float = 0.98) -> pd.DataFrame:
    """ Correlated columns eliminator

    The function drop correlated columns and keep only one of these columns. Usually removing high correlated columns
    gives improvement in model's quality. The task of this function is first to print list of the most correlated
    columns and then remove them by threshold. For more information, please refer to pandas.DataFrame.corr description:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

    :param pd.DataFrame dataframe: Pandas dataframe which contains the dataset e.g. train_dataframe.
    :param bool drop_columns: If true, all correlated columns will be dropped but one.
    :param bool print_columns: If True, information about the correlated columns will be printed to the console.
    :param float threshold: A value between 0 and 1. If the correlation between two columns is larger than this.
                            value, they are considered highly correlated. If drop_columns is True, one of those columns will be dropped. The
                            recommended value of the `threshold` is in [0.7 ... 1].
    
    :return:
            dataframe: A pandas dataframe which contains the dataset after dropping the correlated columns if `drop_columns = True`. Otherwise, the same input dataframe will be returned.
    
    :example:
    
    For checking correlated columns:
    
    >>> dataframe = drop_corr_columns(dataframe, drop_columns=False, print_columns=True, threshold=0.85)
    """

    # 1. calculation
    logger.info("Calculating the correlation matrix")

    correlation_coefficients = dataframe.corr()

    # 2. report
    corr_fields_list = []
    print(f"Columns with correlations more than {str(threshold)} :")
    for i in correlation_coefficients:
        for j in correlation_coefficients.index[correlation_coefficients[i] >= threshold]:
            if i != j and j not in corr_fields_list:
                corr_fields_list.append(j)

                logger.info("Report information if required")

                if print_columns:
                    logger.debug(f"print_columns = {print_columns}: Information should be reported")
                    print(
                        f"{i}-->{j}: r^2={correlation_coefficients[i][correlation_coefficients.index == j].values[0]}"
                    )

    # 3. dropping
    logger.info("Dropping high correlated columns if required")

    if drop_columns:
        logger.debug(f"drop_columns = {drop_columns}: Columns should be dropped")

        print(f"{dataframe.shape[1]} columns total")
        dataframe = dataframe.drop(corr_fields_list, 1)
        print(f"{dataframe.shape[1]} columns left")

    return dataframe


def drop_const_columns(dataframe: pd.DataFrame, drop_columns: bool = True, print_columns: bool = True) -> pd.DataFrame:
    """ Constant value columns eliminator

    This function drops columns that contain constant values. Usually removing constant columns gives improvement in
    model's quality. The task of this function is first to print list of constant columns and then drop them.

    :param pd.DataFrame dataframe: A pandas dataframe that contain the dataset e.g. train_dataframe
    :param bool drop_columns: If true, the columns that contain constant values along all the rows will be dropped.
    :param bool print_columns: If true, information about the columns that contain constant values will be printed to the console

    :return:
            dataframe: A pandas dataframe that contains the dataset after dropping the the columns that contain
            constant values if `drop_columns = True`

    :example:

    For checking the columns which have constant value:

    >>> dataframe = drop_const_columns(dataframe, drop_columns=False, print_columns=True)
    """

    # 1. report
    single_value_cols = []
    for col in dataframe.columns:
        unique_count = dataframe[col].nunique()
        if unique_count < 2:
            single_value_cols.append(col)

            logger.info("Calculating the correlation matrix")

            if print_columns:
                logger.debug(f"print_columns = {print_columns}: Information should be reported")

                print(col, unique_count)

    print(f"Constant columns count: {len(single_value_cols)}")

    # 2. dropping
    logger.info("Dropping high correlated columns if required")

    if drop_columns:
        logger.debug(f"drop_columns = {drop_columns}: Columns should be dropped")

        print(f"{dataframe.shape[1]} columns total")
        dataframe = dataframe.drop(single_value_cols, 1)
        print(f"{dataframe.shape[1]} columns left")

    return dataframe
