import warnings

warnings.filterwarnings('ignore')
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from preprocessing.data_transformer import encode_categorical_features, clean_categorical_features
from preprocessing.data_type_detector import detect_columns_types_summary

# create dataframe for testing the preprocessing functions:
integer_array = np.random.randint(2, size=(10000, 2))
for catagories_numbers in range(5, 50, 10):
    integer_array = np.append(
        integer_array, np.random.randint(catagories_numbers, size=(10000, 2)), axis=1
    )
integer_columns = [f"int_col_{x}" for x in range(integer_array.shape[1])]

continuous_array = np.random.randn(10000, 10)
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
string_arrays = np.random.choice(string_array, (10000, 10))
string_columns = [f"string_col_{x}" for x in range(string_arrays.shape[1])]

integer_dataframe = pd.DataFrame(integer_array, columns=integer_columns)
continuous_dataframe = pd.DataFrame(continuous_array, columns=continuous_columns)
string_dataframe = pd.DataFrame(string_arrays, columns=string_columns)

date_array = np.array([datetime.now() - timedelta(days=x) for x in range(10000)]).reshape(
    10000, 1
)
date_dataframe = pd.DataFrame(date_array, columns=["date_col"])
# print(date_dataframe.dtypes)
train_dataframe = pd.concat(
    [date_dataframe, integer_dataframe, continuous_dataframe, string_dataframe], axis=1
)

dataframe_dict = {
    "train": train_dataframe,
    "valid": train_dataframe,
    "test": train_dataframe,
}
columns_set = detect_columns_types_summary(dataframe_dict)
dataframe_dict_encoded = encode_categorical_features(dataframe_dict, string_columns)
dataframe_dict_not_cleaned = clean_categorical_features(dataframe_dict, string_columns, True)
dataframe_dict_cleaned = clean_categorical_features(dataframe_dict, string_columns, False)


class MyTestCase(unittest.TestCase):
    def test_train_dataset(self):
        self.assertEqual(columns_set["train"]["categorical_integer"], integer_columns)
        self.assertEqual(columns_set["train"]["continuous"], continuous_columns)
        self.assertEqual(columns_set["train"]["categorical_string"], string_columns)
        self.assertEqual(columns_set["train"]["date"], ["date_col"])

    def test_valid_dataset(self):
        self.assertEqual(columns_set["valid"]["categorical_integer"], integer_columns)
        self.assertEqual(columns_set["valid"]["continuous"], continuous_columns)
        self.assertEqual(columns_set["valid"]["categorical_string"], string_columns)
        self.assertEqual(columns_set["valid"]["date"], ["date_col"])

    def test_test_dataset(self):
        self.assertEqual(columns_set["test"]["categorical_integer"], integer_columns)
        self.assertEqual(columns_set["test"]["continuous"], continuous_columns)
        self.assertEqual(columns_set["test"]["categorical_string"], string_columns)
        self.assertEqual(columns_set["test"]["date"], ["date_col"])

    def test_encoding(self):
        for column_i in string_columns:
            self.assertEqual(
                len(dataframe_dict_encoded["train"][column_i].value_counts()),
                len(string_array),
            )

    def test_clean_categorical_features(self):
        for column_i in string_columns:
            self.assertEqual(
                len(dataframe_dict_not_cleaned["train"][column_i].value_counts()),
                len(string_array),
            )

    # TODO currently in tests train=test so checking cleaning is not possible - need to change all previous tests
    def test_clean_categorical_features_cleaned(self):
        self.assertEqual(
            dataframe_dict_cleaned["train"][string_columns].isna().sum().sum(),
            0)


if __name__ == "__main__":
    unittest.main()
