Training
========

.. toctree::
   :maxdepth: 2
   :caption: Parameters Structure Validator:
.. automodule:: training.validator
.. autoclass:: StructureValidation
   :members: __init__, validate_main_keys, validate_data, validate_split, validate_model,
    validate_metrics, validate_predict, features_validator
.. autofunction:: parameters_validator

.. toctree::
   :maxdepth: 2
   :caption: Training:
.. automodule:: training.training
.. autofunction:: train_with_n_split
.. autofunction:: train_with_kfold_cross_validation
.. autofunction:: model_training

.. toctree::
   :maxdepth: 2
   :caption: Hyperparameters Optimizer:
.. automodule:: training.optimizer
.. autofunction:: int_2_float_alpha
.. autofunction:: training_for_optimizing
.. autofunction:: choosing_alpha
.. autofunction:: get_best_alpha_split
.. autofunction:: get_best_alpha_kfold

.. toctree::
   :maxdepth: 2
   :caption: XGboost Model Training:
.. automodule:: training.xgboost_train
.. autofunction:: xgboost_data_preparation
.. autofunction:: xgboost_regression_train
.. autofunction:: xgboost_data_preparation_to_predict
.. autofunction:: training_xgboost_n_split
.. autofunction:: training_xgboost_kfold
.. autofunction:: get_num_round
.. autofunction:: evaluate_xgboost_model

.. toctree::
   :maxdepth: 2
   :caption: Model Evaluator:
.. automodule:: training.classification_model_evaluator
.. autofunction:: classification_evaluate_model
.. autofunction:: classification_model_evaluation
.. automodule:: training.regression_model_evaluator
.. autofunction:: regression_evaluate_model
.. autofunction:: regression_model_evaluation

.. toctree::
   :maxdepth: 2
   :caption: Utilities:
.. automodule:: training.utils
.. autofunction:: read_kfold_config
.. autofunction:: create_model_directory
.. autofunction:: save_model_locally
.. autofunction:: input_parameters_extraction
.. autofunction:: split_dataset
