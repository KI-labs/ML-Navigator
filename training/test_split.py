from training.utils import split_dataset
import numpy as np
import pandas as pd


# create dataframe
def mock_data(number_of_samples):
    integer_array = np.random.randint(2, size=(number_of_samples, 2))
    for catagories_numbers in range(5, 50, 10):
        integer_array = np.append(
            integer_array, np.random.randint(catagories_numbers, size=(number_of_samples, 2)), axis=1
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
split_ratios = [0.2]

sub_datasets = split_dataset(train_dataframe.values, train_target, split_ratios)