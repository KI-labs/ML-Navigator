def unify_dataframes(dataframes_dict: dict, _reference: str, ignore_columns: list) -> dict:
    """ Dataframes unifier

    This function ensures that all datasets have the same features after dropping highly correlated features or columns that have constant values.

    :param dict dataframes_dict: A dictionary that contains Pandas dataframes with nested JSON data type e.g.
                dataframes_dict={ "train": train_dataframe, "test": test_dataframe}
    :param str _reference: The name of the Pandas dataframe that will be used as a reference to adjust the features of other dataframes. Usually it is the train dataframe
    :param list ignore_columns: It contains the columns that should be
                ignored when apply scaling e.g. the id and the target.
    :return:
            - dataframes_dict - A dictionary that contains Pandas dataframes.
    """

    remaining_columns = dataframes_dict[_reference].columns

    for key in dataframes_dict.keys():
        if key != _reference:
            dataframe = dataframes_dict[key]
            try:
                dataframes_dict[key] = dataframe[remaining_columns]
            except Exception as e:
                print(f"The set of remaining columns should be modified. Error: {e}")

                modified_remaining_columns = [x for x in remaining_columns if x not in ignore_columns]
                dataframes_dict[key] = dataframe[modified_remaining_columns]
    return dataframes_dict
