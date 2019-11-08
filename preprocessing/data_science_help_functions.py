import logging
import operator
import os
from collections import Counter
from typing import Tuple, List, Dict, Set, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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


def adversarial_validation(dataframe_dict: dict,
                           ignore_columns: list,
                           max_dataframe_length: int = 100000,
                           threshold: float = 0.7) -> float:
    """ Training a probabilistic classifier to distinguish train/test examples.
    See more info here: http://fastml.com/adversarial-validation-part-one/

    This function tries to check whether test and train data coming from the same data distribution.

    :param dict dataframes_dict: A dictionary that contains pandas dataframes e.g. dataframes_dictionary ={
                'train': train_dataframe, 'test': test_dataframe}
    :param int max_dataframe_length: Max length of dataframe to be considered - make adversarial validation faster
    :param list ignore_columns: List of column to ignore (ID, target, etc...)
    :param float threshold: A value larger than 0 and less than 1. If the result of calculation is greater than threshold - there is sugnificant difference between train and test data

    :return:
            | adversarial_validation_result: Adversarial validation score.
    """

    # Check if it only one dataframe provided
    if len(dataframe_dict) != 2:
        # do nothing and return the original data
        logger.info("Can't apply adversarial_validation because count of dataframes is not equal to 2")
        return None

    # if 2 dataframe than it will be considered as `train` and `test`
    train = dataframe_dict[list(dataframe_dict.keys())[0]]
    test = dataframe_dict[list(dataframe_dict.keys())[1]]

    if len(ignore_columns) > 0:
        columns_to_use = [x for x in list(test.columns) if x not in ignore_columns]
        train = train[columns_to_use]
        test = test[columns_to_use]

    # add identifier and combine
    train['istrain'] = 1
    test['istrain'] = 0

    # max_dataframe_length
    for df in [train, test]:
        if len(df) > max_dataframe_length:
            df = df.head(max_dataframe_length)

    # add identifier and combine
    train['istrain'] = 1
    test['istrain'] = 0
    df_joined = pd.concat([train, test], axis=0)

    # convert non-numerical columns to integers
    df_numeric = df_joined.select_dtypes(exclude=['object'])
    df_obj = df_joined.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_joined = pd.concat([df_numeric, df_obj], axis=1)

    # a new target
    y = df_joined['istrain']
    df_joined.drop('istrain', axis=1, inplace=True)

    # train classifier
    adversarial_validation_result, clf = get_adv_validation_score(df_joined, y)

    # Process result:
    if adversarial_validation_result > threshold:
        print(
            f"WARNING! There is significant difference between {list(dataframe_dict.keys())[0]} and {list(dataframe_dict.keys())[0]}\n"
            f"datasets in terms of feature distribution. Validation score: {adversarial_validation_result}, threshold: {threshold}")
        print(f"Top features are: {xgb_important_features(clf)}")
    else:
        print(
            f"There is no significant difference between {list(dataframe_dict.keys())[0]} and {list(dataframe_dict.keys())[0]}\n"
            f"datasets in terms of feature distribution. Validation score: {adversarial_validation_result}, threshold: {threshold}")
    return None


def get_adv_validation_score(df_joined: pd.DataFrame,
                             y: pd.Series) -> Tuple[float, xgb.sklearn.XGBClassifier]:
    """ Calculate advisarial validation score based on dataframes and XGBClassifier

    :param DataFrame df_joined: Feature dataframe
    :param Series y: Target series

    :return:
            | clf: Trained model
            | mean of KFold validation results (ROC-AUC scores)
    """

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=44)
    xgb_params = {
        'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9,
        'colsample_bytree': 0.9, 'objective': 'binary:logistic',
        'silent': 1, 'n_estimators': 10, 'gamma': 1,
        'min_child_weight': 4
    }
    clf = xgb.XGBClassifier(**xgb_params, seed=10)
    results = []
    logger.info('Adversarial validation checking:')
    for fold, (train_index, test_index) in enumerate(skf.split(df_joined, y)):
        fold_xtrain, fold_xval = df_joined.iloc[train_index], df_joined.iloc[test_index]
        fold_ytrain, fold_yval = y.iloc[train_index], y.iloc[test_index]
        clf.fit(fold_xtrain, fold_ytrain, eval_set=[(fold_xval, fold_yval)],
                eval_metric='logloss', verbose=False, early_stopping_rounds=10)
        fold_ypred = clf.predict_proba(fold_xval)[:, 1]
        fold_score = roc_auc_score(fold_yval, fold_ypred)
        results.append(fold_score)
        logger.info(f"Fold: {fold + 1} shape: {fold_xtrain.shape} score: {fold_score}")

    return round(np.mean(results), 2), clf


def xgb_important_features(xgb: xgb.sklearn.XGBClassifier,
                           top_features: int = 5) -> str:
    """ Get top of the most important features from a trained model

    :param XGBClassifier xgb: A trained model
    :param int top_features: Max length of features to send back

    :return:
            | A string with a list of the most important features plus their importance
    """

    # get features
    feat_imp = xgb.get_booster().get_score(importance_type='gain')

    # round importances
    for dict_key in feat_imp:
        feat_imp[dict_key] = round(feat_imp[dict_key])

    # sort by importances
    sorted_x = sorted(feat_imp.items(), key=operator.itemgetter(1))
    sorted_x.reverse()

    return str(list(sorted_x[:top_features]))
