import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from feature_engineering.feature_generator import one_hot_encdoing_sklearn
# create dataframe for testing the preprocessing functions:

integer_array = np.random.randint(2, size=(1000, 1))
for catagories_numbers in range(5, 50, 25):
    integer_array = np.append(
        integer_array, np.random.randint(catagories_numbers, size=(1000, 1)), axis=1
    )
integer_columns = [f"int_col_{x}" for x in range(integer_array.shape[1])]

continuous_array = np.random.randn(1000, 5)
continuous_columns = [f"cont_col_{x}" for x in range(continuous_array.shape[1])]

string_array = [
    "pizza",
    "ball",
    "star3",
    "car",
    "01515",
    "cat75",
    "moon",
    "lol",
    "dddd",
    "wtf",
    "xXx",
    "82008",
    "mailbox",
]
string_arrays = np.random.choice(string_array, (1000, 1))
string_columns = [f"string_col_{x}" for x in range(string_arrays.shape[1])]

integer_dataframe = pd.DataFrame(integer_array, columns=integer_columns)
continuous_dataframe = pd.DataFrame(continuous_array, columns=continuous_columns)
string_dataframe = pd.DataFrame(string_arrays, columns=string_columns)

date_array = np.array([datetime.now() - timedelta(days=x) for x in range(1000)]).reshape(
    1000, 1
)
date_dataframe = pd.DataFrame(date_array, columns=["date_col"])
# print(date_dataframe.dtypes)
train_dataframe = pd.concat(
    [date_dataframe, integer_dataframe, continuous_dataframe, string_dataframe], axis=1
)

dataframes_dict = {
    "train": train_dataframe,
    #"valid": train_dataframe,
    #"test": train_dataframe,
}

categorical_feature = ["int_col_0", "int_col_1", "int_col_2", "string_col_0"]
class_number_range = [3, 50]
ignore_columns = []

encoded_data = one_hot_encdoing_sklearn(dataframes_dict, "train", categorical_feature, class_number_range,
                                        ignore_columns)

number_of_columns_encoded_data = encoded_data["train"].shape[1]
number_of_cat = len(train_dataframe["int_col_1"].value_counts()) + len(train_dataframe["int_col_2"].value_counts()) +\
                len(train_dataframe["string_col_0"].value_counts()) - 3 + train_dataframe.shape[1]

class MyTestCase(unittest.TestCase):

    def test_valid_dataset(self):
        self.assertEqual(number_of_cat, number_of_columns_encoded_data)


if __name__ == "__main__":
    unittest.main()
