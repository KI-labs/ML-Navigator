def unify_dataframes(dataframes_dict: dict, _reference: str, ignore_columns: list) -> dict:
    """ Dataframes unifier

    :param dataframes_dict:
    :param _reference:
    :param ignore_columns:
    :return:
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
