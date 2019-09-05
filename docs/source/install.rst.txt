Install / Quick Start
=====================

Quick Start
-----------

To install the ML-Navigator Package you need to have `Python 3.6`:

1. clone the git repository:

.. code:: shell

        git clone https://github.com/KI-labs/ML-Navigator.git

.. code:: shell

        cd ML-Navigator


2. create a directory under the name "data" and move your data files to it e.g. "train.csv" and "test.csv"

3. create a virtual environment:

.. code:: shell

        pip install virtualenv

        virtualenv venv

        source /venv/bin/activate

After setting up the virtual environment, you can install the package using pip command as follows:

.. code:: shell

        pip install .

4. OR install the required packages

.. code:: shell

        pip install -r requirements.txt

5. Create a directory under the name "data" inside the project root directory.


6. To run the tutorials, you can download the "train.csv" and "test.csv" datasets from Kaggle website:

    * inside `./data/flow_0` and `./data/flow_1` store the data from the the House Prices - Advanced Regression Techniques competition:
        https://www.kaggle.com/c/house-prices-advanced-regression-techniques

    * inside `./data/flow_2` directory store the data from the the TMDB Box Office Prediction competition:
        https://www.kaggle.com/c/tmdb-box-office-prediction

File Structure
--------------

The structure of the directories looks like the following:

.. code:: shell

        .
    ├── LICENSE
    ├── setup.py
    ├── MANIFEST.in
    ├── data
    │   ├── flow_0
    │   ├── flow_1
    │   ├── flow_2
    │   └── flow_3
    ├── feature_engineering
    │   ├── __init__.py
    │   ├── feature_generator.py
    │   └── test.py
    ├── flows
    │   ├── __init__.py
    │   ├── example.yaml
    │   ├── flow_0.drawio
    │   ├── flow_0.json
    │   ├── flow_1.drawio
    │   ├── flow_1.json
    │   ├── flow_2.drawio
    │   ├── flow_2.json
    │   ├── flow_3.drawio
    │   ├── flow_3.json
    │   ├── flows.py
    │   ├── utils.py
    │   ├── text_helper.py
    │   └── yaml_reader.py
    ├── images
    │   ├── flow_0_record_middle_size.gif
    │   └── logo.png
    ├── prediction
    │   ├── __init__.py
    │   └── model_predictor.py
    ├── preprocessing
    │   ├── README.md
    │   ├── __init__.py
    │   ├── data_clean.py
    │   ├── data_explorer.py
    │   ├── data_science_help_functions.py
    │   ├── data_transformer.py
    │   ├── data_type_detector.py
    │   ├── json_preprocessor.py
    │   ├── test_loading_data.py
    │   ├── test_preprocessing.py
    │   └── utils.py
    ├── readme.md
    ├── requirements.txt
    ├── training
    │   ├── __init__.py
    │   ├── model_evaluator.py
    │   ├── optimizer.py
    │   ├── test_split.py
    │   ├── test_training.py
    │   ├── training.py
    │   ├── utils.py
    │   └── validator.py
    ├── tutorials
    │   ├── flow_0.png
    │   ├── flow_0.ipynb
    │   ├── flow_1.ipynb
    │   ├── flow_1.png
    │   ├── flow_2.ipynb
    │   ├── flow_2.png
    │   ├── flow_3.ipynb
    │   └── flow_3.png
    ├── venv
    └── visualization
        ├── __init__.py
        └── visualization.py
