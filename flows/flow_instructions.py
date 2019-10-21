from blessings import Terminal

# For colorful and beautiful formatted print()
term = Terminal()


class FlowInstructions:

    @staticmethod
    def read_data(specific_flow: int = 0,
                  specific_directory: str = "./data",
                  specific_file_list: str = "'train.csv','test.csv'"):
        print(term.bold(term.magenta("Please use the following function to read the data")))
        print(term.green_on_black("dataframe_dict, columns_set = flow.load_data(path : str, files_list : list)"))
        print(term.bold(term.magenta("For example: ") + term.green_on_black(f"path = {specific_directory}")))
        print(term.bold(term.magenta(
            "If your data is in a nested directory, it is better to os.path.join. For example:\n") + term.green_on_black(
            f"path = os.path.join('data', 'flow_{specific_flow}')")))
        print(term.bold(term.magenta("For example: ") + term.green_on_black(f"files_list = [{specific_file_list}]")))
        print(term.bold(term.magenta("The output is a dictionary that contains dataframes e.g.\n")))
        print(term.blue("dataframe_dict = {'train': train_dataframe,'test': test_dataframe}"))

    @staticmethod
    def encode_categorical_features():
        print(term.bold(term.magenta("If you have categorical features with string labels, Encode the categorical "
                                     "features by applying the following function:\n") + term.green_on_black(
            "dataframe_dict, columns_set = flow.encode_categorical_feature(dataframe_dict: dict)")))

    @staticmethod
    def scale_numeric_features():
        print(term.bold(term.magenta("If you have numeric features, it is a good idea to normalize numeric features." +
                                     "Use the following function for feature normalization:\n") +
                        term.green_on_black(
                            " dataframe_dict, columns_set = flow.scale_data (dataframe_dict:"
                            " dict, ignore_columns: list)")))
        print(term.bold(term.magenta("For example: ") + term.green_on_black("ignore_columns = ['id', 'target']")))

    @staticmethod
    def train_a_model():
        print(term.bold(term.magenta("Your features are ready to train the model: ")))
        print(term.bold(term.magenta("You can start training the model by applying the following function: ")))
        print(term.green_on_black("model_index_list, save_models_dir, y_test = flow.training(parameters)"))
        print('parameters = {\n'
              '     "data": {\n'
              '         "train": {"features": train_dataframe, "target": train_target}, \n'
              '         "valid": {"features": valid_dataframe, "target": valid_target}, \n'
              '         "test": {"features": test_dataframe, "target": test_target}, \n'
              '     }, \n'
              '     "split": {\n'
              '         "method": "split",\n'
              '         "split_ratios": 0.2\n'
              '         }, \n '
              '     "model": {"type": "Ridge linear regression",\n'
              '               "hyperparameters": {"alpha": 1,  # alpha:optimize}\n'
              '             }, \n '
              '     "metrics": ["r2_score", "mean_squared_error"]\n'
              '}')

    @staticmethod
    def one_hot_encoding():
        print(term.bold(term.magenta(
            "You have categorical features. Apply one-hot encoding to the categorical features by applying the"
            " following function:\n") + term.green_on_black('dataframe_dict, columns_set = flow.features_encoding('
                                                            '"one-hot", dataframe_dict: dict, reference: str,'
                                                            ' ignore_columns: list, class_number_range = [3, 50])')))
        print(term.bold(term.magenta("Since one-hot encoding can produce a lot of features, class_number_range will "
                                     "limit the encoding process only for features which have between 3 and 49"
                                     " unique values.")))
        print(term.bold(term.magenta("If you are solving a classification problem, you should exclude the target from "
                                     "the one - hot encoding process by defining the ignore_columns\n ")
                        + term.green_on_black(" ignore_columns = [ < your target / label >]\n ")
                        + term.magenta("You can add more columns to the ignore_columns list to ignore")))

    @staticmethod
    def flatten_JSON_data():
        print(term.bold(term.magenta(
            "You have JSON nested data inside the dataframe columns. "
            "Flatten the nested JSON data by applying the following function:\n") + term.green_on_black(
            "dataframe_dict, columns_set= flow.flatten_json_data(dataframe_dict)")))

    @staticmethod
    def drop_high_corrected_columns_but_keep_one():
        print(term.bold(term.magenta(
            "If some features are highly correlated, they do not provide more information about the target prediction. "
            "It is a good idea to drop such features but keep one:\n") + term.green_on_black(
            "dataframe_dict, columns_set = flow.drop_correlated_columns(dataframe_dict: dict, ignore_columns: list)") +
                        term.magenta("An example of the ignore_columns list: \n")
                        + term.green_on_black(" ignore_columns = [target]\n")))

    @staticmethod
    def delete_features_with_constant_values():
        print(term.bold(term.magenta(
            "If some features have in all rows the same value, they have no influence on the target prediction. "
            "It is a good idea to delete such features:\n") + term.green_on_black(
            "dataframe_dict, columns_set = flow.drop_columns_constant_values("
            "dataframe_dict: dict, ignore_columns: list)") +
                        term.magenta("An example of the ignore_columns list: \n")
                        + term.green_on_black(" ignore_columns = [target]\n")))

    @staticmethod
    def explore_data():
        print(term.bold(term.magenta("If you want to explore the data you can run one of the following functions: ")))
        print(term.bold(term.magenta("1 . ") + term.green_on_black(
            "flow.exploring_data(dataframe_dict: dict, key_i: str)")))
        print(term.bold(term.magenta("For example: ") + term.green_on_black(
            "flow.exploring_data(dataframe_dict, 'train')")))
        print(term.bold(term.magenta("2 . ") + term.green_on_black(
            "flow.comparing_statistics(dataframe_dict: dict)")))
        print(term.bold(term.magenta("For example: ") + term.green_on_black(
            "flow.comparing_statistics(dataframe_dict)")))
        print("\n")

    @staticmethod
    def target_based_categorical_feature_encoding():
        print(term.bold(term.magenta("Apply target-based encoding to the categorical features by applying the"
                                     " following function:\n")))
        print(term.bold(term.green_on_black(
            'dataframe_dict, columns_set = flow.features_encoding("target", dataframe_dict: dict, reference: str,'
            ' ignore_columns: list, target: str)')))
        print(term.magenta("An example of the ignore_columns list: \n") +
              term.green_on_black(" ignore_columns = ['id', 'target']\n"))
        print(term.magenta("An example of the reference: \n") +
              term.green_on_black(" reference = 'train'\n"))

    @staticmethod
    def frequency_based_categorical_feature_encoding():
        print(term.bold(term.magenta("Apply target-based encoding to the categorical features by applying the"
                                     " following function:\n")))
        print(term.bold(term.green_on_black(
            'dataframe_dict, columns_set = flow.features_encoding("frequency",'
            ' dataframe_dict: dict, reference: str, drop_encoded_features = False)')))
        print(term.magenta("An example of the ignore_columns list: \n") +
              term.green_on_black(" ignore_columns = ['id', 'target']\n"))
        print(term.magenta("An example of the reference: \n") +
              term.green_on_black(" reference = 'train'\n"))
