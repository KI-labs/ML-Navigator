import logging
import os
from typing import Tuple, List, Dict, Set, Union
from collections import Counter

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


def detect_id(dataframes_dictionary: dict) -> Union[set, str]:
    """ ID candidates detector

    the following assumptions are considered:

    | 1. Id exists in all the loaded datasets;
    | 2. All the id' values are unique in all datasets;
    | 3. There are no missing values.

    If the function finds more than one feature that satisfies the assumptions, it suggests a list of candidates
    and the user has to decide which one is the id

    :param dict dataframes_dictionary: It is a dictionary that contains pandas dataframes e.g. dataframes_dictionary ={
    'train': train_dataframe, 'test': test_dataframe}

    :return A set of strings values as id candidates.

    :rtype: set
    """

    id_candidates = []

    for key_i, dataframe in dataframes_dictionary.items():
        for column_i in dataframe.columns:
            if len(dataframe[column_i].value_counts()) == dataframe[column_i].count() and \
                    len(dataframe[column_i].value_counts()) == dataframe.shape[0]:
                logger.info(f"{column_i} found to be a candidate as an ID")

                id_candidates.append(column_i)

    id_candidates = set(id_candidates)

    # the number 5 in the if statement should be defined as variable later that can be added by the user
    if len(id_candidates) > 5:
        logger.info("Too many options for ids.")
        return "Too many options for ids. Not possible to detect id"
    elif len(id_candidates) == 0:
        logger.info("No ids candidates were found")
        return "No ids candidates were found"

    logger.info("Detecting the id is finished")

    return set(id_candidates)


def detect_target(dataframes_dictionary: dict) -> Union[list, str]:
    """ Target candidates detector

    The following assumptions are considered:

    | 1. Target is missing in at least one dataset-which is the test dataset usually;
    | 2. The target has no missing values.

    If the function finds more than one feature that satisfies the assumptions, it suggests a list of candidates
    and the user has to decide which one is the target

    :param dict dataframes_dictionary: It is a dictionary that contains pandas dataframes e.g. dataframes_dictionary ={
    'train': train_dataframe, 'test': test_dataframe}

    :return
        target_candidates_2: A list of candidates as strings that satisfies the two mentioned assumptions above

    :rtype: list

    """

    target_candidates_1 = []
    target_candidates_2 = []
    columns = []

    for key_i, dataframe in dataframes_dictionary.items():
        # Getting all columns names of all datasets in one place
        columns = columns + list(dataframe.columns)

    # check the first assumptions: calculating the occurrence
    try:
        occurrence = Counter(columns)
        minimum_occurrence = min(occurrence.values())
        target_candidates_1 = [x for x, y in occurrence.items() if y == minimum_occurrence]
        logger.info("Checking the first assumption (occurrence) for finding the target is done!")

    except Exception as e:
        logger.error(f"Error: {e}")

    # check the second assumption: No missing values
    for column_i in target_candidates_1:
        for key_i, dataframe in dataframes_dictionary.items():
            if column_i in dataframe.columns:
                if len(dataframe[column_i]) - dataframe[column_i].count() == 0:
                    target_candidates_2.append(column_i)

    logger.info("Checking the second assumption (missing values) for finding the target is done!")

    if len(target_candidates_2) == 0:
        logger.info("No target was detected")
        return "No target was detected"
    elif len(target_candidates_2) > 5:
        logger.info("Too many options for target")
        return "Too many options for target. Not possible to detect target"

    logger.info("Detecting the target is finished")

    return target_candidates_2


def detect_problem_type(dataframes_dictionary: dict,
                        target_candidates: list,
                        threshold: float = 0.1) -> Union[dict, str]:
    """ Problem type detector

    This function tells what type of problem that should be solved: classification or regression

    :param dict dataframes_dictionary: A dictionary that contains pandas dataframes e.g. dataframes_dictionary ={
                                        'train': train_dataframe, 'test': test_dataframe}
    :param list target_candidates: It is list of string of the possible target candidates
    :param float threshold: A value larger than 0 and less than 1. It defines when the problem is considered as a
    regression  problem or classification problem

    :return:
            problem_type: dictionary of target candidates associated with the problem type

    :rtype: dict
    """

    problem_type = {}

    if isinstance(target_candidates, list) and len(target_candidates) > 0:
        for column_i in target_candidates:
            for key_i, dataframe in dataframes_dictionary.items():
                if column_i in dataframe.columns:
                    if len(dataframe[column_i].value_counts()) / dataframe.shape[0] < threshold:
                        problem_type[column_i] = "classification"
                        logger.debug(f"For the target {column_i}: classification")
                    else:
                        problem_type[column_i] = "regression"
                        logger.debug(f"For the target {column_i}: regression")

    else:
        logger.info("No valid target candidates")
        return "No problem type to detect"

    logger.info("Detecting the problem type is finished")

    return problem_type


def detect_id_target_problem(dataframes_dict: dict, threshold: float = 0.1) -> Tuple[Set, List, Dict]:
    """ ID Target Problem type detector

    This function tries to find which column is the ID and which one is the target and what type of the problem
    to be solved. It uses `detect_id`, `detect_target` and `detect_problem_type` functions.

    :param dict dataframes_dict: A dictionary that contains pandas dataframes e.g. dataframes_dictionary ={
                'train': train_dataframe, 'test': test_dataframe}
    :param float threshold: A value larger than 0 and less than 1. It defines when the problem is a regression
            problem or classification problem

    :return:
            | possible_ids: A list of candidates as strings.
            | possible_target: list of candidates as strings.
            | possible_problems: dictionary of target candidates associated with the problem type.
    """

    possible_ids = detect_id(dataframes_dict)
    possible_target = detect_target(dataframes_dict)
    possible_problems = detect_problem_type(dataframes_dict, possible_target, threshold=threshold)

    logger.info("Running all the process to detect id, target and the problem type is finished")

    # Showing the info to the user
    print(f"The possible ids are:\n {possible_ids}")
    print(f"The possible possible_target are:\n {possible_target}")
    print(f"The type of the problem that should be solved:\n {possible_problems}")
    return possible_ids, possible_target, possible_problems
