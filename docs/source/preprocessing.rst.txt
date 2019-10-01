Preprocessing
=============

.. toctree::
   :maxdepth: 2
   :caption: Data Type Detector:
.. automodule:: preprocessing.data_type_detector
.. autoclass:: ColumnDataFormat
   :members: __init__, find_date_columns, number_or_string, json_detector, categorical_or_numeric
.. autofunction:: detect_column_types
.. autofunction:: detect_columns_types_summary

.. toctree::
   :maxdepth: 2
   :caption: Data Transformer:
.. automodule:: preprocessing.data_transformer
.. autofunction:: standard_scale_numeric_features
.. autofunction:: encoding_categorical_feature
.. autofunction:: encode_categorical_features

.. toctree::
   :maxdepth: 2
   :caption: Data Cleaning:
.. automodule:: preprocessing.data_clean
.. autofunction:: drop_corr_columns
.. autofunction:: drop_const_columns

.. toctree::
   :maxdepth: 2
   :caption: Data Explorer:
.. automodule:: preprocessing.data_explorer
.. autofunction:: print_repeated_values
.. autofunction:: explore_data
.. autoclass:: ExploreData
   :members: __init__, data_explore

.. toctree::
   :maxdepth: 2
   :caption: JSON Preprocessor:
.. automodule:: preprocessing.json_preprocessor
.. autofunction:: extract_json_from_list_dict
.. autofunction:: extract_json_objects
.. autofunction:: normalize_feature
.. autofunction:: apply_normalize_feature
.. autofunction:: column_validation
.. autofunction:: combine_new_data_to_original
.. autofunction:: feature_with_json_detector
.. autofunction:: combine_columns
.. autofunction:: flat_json

.. toctree::
   :maxdepth: 2
   :caption: Utilities:
.. automodule:: preprocessing.utils
.. autofunction:: read_data



