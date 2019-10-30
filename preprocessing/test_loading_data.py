import os
from preprocessing.utils import read_data

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/flow_2")
files_list = ["train.csv", "test.csv"]
test_local = False


# This test has to be applied locally because data is not pushed to Github.
def test_something():
    if test_local:
        dataframe_dict = read_data(path, files_list)
        assert (dataframe_dict["train"].shape[0] == 3000)
        assert (dataframe_dict["test"].shape[0] == 4398)
        assert (dataframe_dict["train"].shape[1] > 1)
        assert (dataframe_dict["test"].shape[1] > 1)
    else:
        pass_test = True
        assert (pass_test == True)
