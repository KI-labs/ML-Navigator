from training.validator import parameters_validator
import numpy as np
import pandas as pd


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

# This should not raise any exception
parameters1 = {
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

# This should not raise any exception
parameters2 = {
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
    "metrics": ["accuracy_score", "mean_squared_error"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}

# if the name of the metrics is written wrong, this should raise a value error.
parameters3 = {
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
    "metrics": ["accuracy_scores", "roc_auc_score"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}

# If the features inside the data.train is not pandas dataframe, this should raise a TypeError
parameters4 = {
    "data": {
        "train": {"features": train_dataframe.values, "target": train_target},
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

# If the predict key exists and the features inside the predict.test is not pandas dataframe,
# this should raise a TypeError
parameters5 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 0.2,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
    },
    "model": {"type": "Ridge linear regression",
              "hyperparameters": {"alpha": 1,  # alpha:optimize
                                  },
              },
    "metrics": ["r2_score", "mean_squared_error"],
    "predict": {
        "test": {"features": test_dataframe.values}
    }
}

# If the target inside the data.train is not a numpy, this should raise a TypeError
parameters6 = {
    "data": {
        "train": {"features": train_dataframe, "target": pd.DataFrame(train_target)},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 0.2,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the predict key is missing, this should not be a problem
parameters7 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 0.2,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
    },
    "model": {"type": "Ridge linear regression",
              "hyperparameters": {"alpha": 1,  # alpha:optimize
                                  },
              },
    "metrics": ["r2_score", "mean_squared_error"],
}

# If the split_ratios inside split key is not in ]0,1[, a ValueError should be raised
parameters8 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 1.5,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the split_ratios inside split key is not in ]0,1[, a ValueError should be raised
parameters9 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": 0,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the split_ratios inside split key is not in ]0,1[, a ValueError should be raised
parameters10 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": (0.1, 5),  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the split_ratios inside split key is not in ]0,1[, a ValueError should be raised
parameters11 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": (5, 0.5),  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If one of the data features or target are missing, it should raise a TypeError
parameters12 = {
    "data": {
        "train": {"target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "split",  # "method":"kfold"
        "split_ratios": (0.5, 0.5),  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the split method is kfold, we should have foldnr as the second key. Otherwise it raises a TypeError
parameters13 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "split_ratios": (0.5, 0.5),  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the split method is kfold, foldnr should be integer >=1. Otherwise, ValueError should be raised
parameters14 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "fold_nr": 0.1,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
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

# If the type of the model should not be missing. Otherwise, TypeError should be raised
parameters15 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "fold_nr": 5,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
    },
    "model": {
        "hyperparameters": {"alpha": 1,  # alpha:optimize
                            },
    },
    "metrics": ["r2_score", "mean_squared_error"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}

# If hyperparameters should be a dict
parameters16 = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target},
        "valid": {"features": valid_dataframe, "target": valid_target},
        "test": {"features": test_dataframe, "target": test_target},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "fold_nr": 5,  # foldnr:5 , "split_ratios": 0.2 # "split_ratios":(0.3,0.2)
    },
    "model": {"type": "Ridge linear regression",
              "hyperparameters": [1,  # alpha:optimize
                                  ],
              },
    "metrics": ["r2_score", "mean_squared_error"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}

# positive tests
assert parameters_validator(parameters1)
assert parameters_validator(parameters2)
assert parameters_validator(parameters7)

# negative tests
try:
    assert not parameters_validator(parameters3)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters4)
except TypeError:
    assert True

try:
    assert not parameters_validator(parameters5)
except TypeError:
    assert True

try:
    assert not parameters_validator(parameters6)
except TypeError:
    assert True

try:
    assert not parameters_validator(parameters8)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters9)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters10)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters11)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters12)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters13)
except TypeError:
    assert True

try:
    assert not parameters_validator(parameters14)
except ValueError:
    assert True

try:
    assert not parameters_validator(parameters15)
except TypeError:
    assert True

try:
    assert not parameters_validator(parameters16)
except TypeError:
    assert True
