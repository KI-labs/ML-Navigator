import unittest
import numpy as np
import pandas as pd

from feature_engineering.feature_generator import decode_features_with_appearance_frequency, encoding_features


def create_data():
    # create dataframe for testing the frequency-based decoder functions:
    integer_array = np.array([1, 1, 23, 2, 3, 3, 3, 5])
    integer_columns = ["int_col"]

    integer_dataframe = pd.DataFrame(integer_array, columns=integer_columns)
    return integer_dataframe


class EncodingTest(unittest.TestCase):

    def test_frequency_decoding(self):
        # test 1
        dataframe_dict = {"train": create_data()}
        reference = 'train'
        categorical_features = ["int_col"]
        decoded_dataframe_dict = decode_features_with_appearance_frequency(dataframe_dict, reference,
                                                                           categorical_features)
        decoded_dataframe = decoded_dataframe_dict["train"]
        decoded_array = decoded_dataframe["int_col_frequency_encoding"].values
        expected_results = np.array([2 / 8, 2 / 8, 1 / 8, 1 / 8, 3 / 8, 3 / 8, 3 / 8, 1 / 8])
        self.assertTrue((expected_results== decoded_array).all)

    def test_one_hot_encoding(self):
        # test 2
        dataframe_dict = {"train": create_data()}
        encoding_type = "one-hot"
        reference = "train"
        categorical_features = ["int_col"]
        ignore_columns = []
        one_hot_encoding_dict = encoding_features(encoding_type, dataframe_dict, reference, categorical_features,
                                                  ignore_columns)
        one_hot_encoding_array = one_hot_encoding_dict["train"].values
        expected_results_one_hot_list = np.array([[1, 0, 0, 0, 0],
                                                  [1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0]])
        self.assertTrue((expected_results_one_hot_list==one_hot_encoding_array).all())

    def test_target_encoding(self):
        # test 3
        dataframe = create_data()
        target = [0, 1, 0, 0, 1, 1, 0, 1]
        dataframe["target"] = target
        dataframe_dict = {"train": dataframe}
        encoding_type = "target"
        reference = "train"
        categorical_features = ["int_col"]
        ignore_columns = ["target"]
        target_encoding_dict = encoding_features(encoding_type, dataframe_dict, reference, categorical_features,
                                                 ignore_columns, target_name="target")
        target_encoding_dataframe = target_encoding_dict["train"]
        expected_results_target = np.array([0.5000, 0.5000, 0.5000, 0.5000, 0.6468, 0.6468, 0.6468, 0.5000])

        self.assertTrue((np.array(round(target_encoding_dataframe["col_target_encoding_0"], 4))==
                                 expected_results_target).all())


if __name__ == '__main__':
    unittest.main()
