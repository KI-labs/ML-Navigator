import unittest
import numpy as np
import pandas as pd
from training.xgboost_train import xgboost_data_preparation, xgboost_regression_train


# create dataframe for testing the preprocessing functions:
def mock_data(number_of_samples):
    integer_array = np.random.randint(2, size=(number_of_samples, 2))
    for categories_numbers in range(5, 50, 10):
        integer_array = np.append(
            integer_array, np.random.randint(categories_numbers, size=(number_of_samples, 2)), axis=1
        )
    integer_columns = [f"int_col_{x}" for x in range(integer_array.shape[1])]

    continuous_array = np.random.randn(number_of_samples, 10)
    continuous_columns = [f"cont_col_{x}" for x in range(continuous_array.shape[1])]

    integer_dataframe = pd.DataFrame(integer_array, columns=integer_columns)
    continuous_dataframe = pd.DataFrame(continuous_array, columns=continuous_columns)

    dataframe = pd.concat([integer_dataframe, continuous_dataframe], axis=1)
    target = (np.sum(continuous_array, axis=1) - 1) / (1 + np.sum(integer_array, axis=1))
    return dataframe, target


train_dataframe, train_target = mock_data(1000)
valid_dataframe, valid_target = mock_data(100)
test_dataframe, test_target = mock_data(100)

parameters_linear = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 0.2,  # foldnr:5 , "split_ratios": 0.8 # "split_ratios":(0.7,0.2)
    },
    "model": {"type": "Ridge linear regression",
              "hyperparameters": {"alpha": 1,  # alpha:optimize
                                  },
              },
    "metrics": ["r2_score", "mean_squared_error"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}

validation_list = []
for data_i in parameters_linear["data"].keys():
    dataframe = parameters_linear["data"][data_i]["features"]
    target = parameters_linear["data"][data_i]["target"]

    validation_list = xgboost_data_preparation(validation_list, dataframe, target, data_i)

assert isinstance(validation_list, list)
assert len(validation_list) == 3

hyperparameters = {'max_depth': 5, 'eta': 1, 'eval_metric': "rmse"}
bst = xgboost_regression_train(validation_list, hyperparameters, num_round=10)
print(type(bst))
assert isinstance(bst, object)
