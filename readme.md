<p align="center">
  <img src="./images/logo.png" alt="Logo of ML-Navigator"/>
</p>

For full documentation please check [ML-Navigator](https://ki-labs.github.io/ML-Navigator/index.html)


[![Build Status](https://travis-ci.com/KI-labs/ML-Navigator.svg?token=qeb22MSpJyy4b3twFCDG&branch=master)](https://travis-ci.com/KI-labs/ML-Navigator)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Concept

ML-Navigator is a tutorial-based Machine Learning framework. The main component of ML-Navigator is the flow.
A flow is a collection of compact methods/functions that can be stuck together with guidance texts. 

The flow functions as a map which shows the road from point A to point B. The guidance texts function as navigator 
instructions that help the user to figure out the next step after executing the current step.

Like the car navigator, the user is not forced to follow the path. At any point, the user can take a break to explore 
data, modify the features and make any necessary changes. The user can always come back to the main path which the flow 
defines.

> The flows are created by the community for the community.
<p align="center">
    <img src="./images/flow_0_record_middle_size.gif" alt="flow concept" width="720px" align="middle"/>
</p>

# Introduction


> ML-Navigator is a free editable maps collection of the data science world.<br>

## Why ML-Navigator

ML-Navigator standardizes exchanging knowledge among data scientists. Junior data scientists
can learn and apply data science best-practice in a white-box mode. Senior data scientists can
automate many repetitive processes and share experience effectively. Enterprises can standardize
data science among different departments.

## Background
Data Science has been attracting smart people who have different backgrounds, 
experiences, and field knowledge. The transformation journey from other disciplines into 
data science is not a straight forward process. It is time and effort consuming based on 
the motivation, the background, and the industry where the data scientist wants to work.
The data science new joiners follow multiple paths to sharp their data science skills. 
Some of these paths are:<br>
- Online courses: Some E-learning platforms, e.g., LinkedIn-learning, 
provide practical courses to solve specific data science problems. 
Other platforms, such as Coursera, offer theory-based courses. In LinkedIn-learning 
platform alone, there are 400+ courses related to data science. Selecting the best 
courses among all of those numerous courses is a challenge for newbies. Moreover, 
the theory-based courses require solid mathematical knowledge, especially in 
calculus and linear algebra.
- Data Science online platforms: Such platforms, like Kaggle, offer playground or 
prize-based competitions. Junior data scientists can learn a lot by applying their 
knowledge and reading kernels, which data scientists write to share their experience. 
Poorly written code and lack of documentation can be frustrating for newbies who want to 
learn what happened behind the scenes.
- Manuals of well-known data science frameworks: There are many open-source frameworks 
which provide an industry-proven implementation of many methods that have been used by 
data scientists. Many of these frameworks don't share the same syntax. Data scientists 
may need to learn new syntax each time they switch to a new framework.
- Learning from senior data scientists: Onboarding junior data scientists may require 
time which senior data scientists don't always have.


## To whom is ML-Navigator
### Data science new joiners
A new joiner is a person who wants to move into data science from a different discipline.
A new joiner can also be a person who wants to be a part of the data team but not
 a full-time data scientist, e.g., developers with sufficient coding skills.
ML-Navigator provides the data science new joiners the path to analyze real data.
It helps the user to navigate through predefined flows, which are End-2-End data
 science pipelines. The user can load a specific flow and follow the instructions starting
 from reading data until training the model. 
The user can start with the most straightforward flow and later use more complicated 
flows to train accurate models if needed.

### Senior data scientists
Experienced data scientists may be interested in automating many processes that they follow
 frequently. They can build a flow for each specific problem type. 
 The flow can be created from scratch or by modifying or combining other flows. 
 They can share their flows with the community and exchange their experience with other data 
 scientists.


### ML-Navigator for enterprises
ML-navigator can standardize the data science experience in large enterprises.
Junior data scientists can be productive and efficient from the first day. The onboarding process
can be fast, concrete, but not abstracted.

Data scientists may use AutoML to produce multiple types of models as an alternative to digging deep in data and gaining 
new knowledge. AutoML can create a large number of models. 
However, it doesn't guarantee that the user gets the model that satisfies the quality requirements. 
It needs a long time for testing a wide range of hyperparameters values. 
Model reproducibility can be an issue when creating models using AutoML.

# How-to Guides
## How to install ML-Navigator

To install the ML-Navigator Package you need to have `Python 3.6`:

You can install ML-Navigator using the `pip` tool directly:

`pip install ML-Navigator`

To install the ML-Navigator Package from the Github repo:

1. clone the git repository:

    `$ git clone https://github.com/KI-labs/ML-Navigator.git`<br>
    `$ cd ML-Navigator`<br>
    

2. create a directory under the name "data" and move your data files to it e.g. "train.csv" and "test.csv"

3. create a virtual environment

    `$ pip install virtualenv`<br>
    `$ virtualenv venv`<br>
    `$ source /venv/bin/activate`
    
4. After setting up the virtual environment, you can install the package using pip command as follows:

   `$ pip install .`<br>
    
## File Structure
The structure of the directories looks like the following

````
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

````

# Tutorials

Create a directory under the name "data" inside the project root directory.

To run the tutorials, you can download the "train.csv" and "test.csv" datasets from Kaggle website:

    * inside `./data/flow_0` and `./data/flow_1` store the data from the [House Prices: Advanced Regression Techniques competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

    * inside `./data/flow_2` directory store the data from the: [TMDB Box Office Prediction competition](https://www.kaggle.com/c/tmdb-box-office-prediction)

Please check the tutorials directory: [Tutorials](https://github.com/KI-labs/ML-Navigator/tree/master/tutorials)

# Reference

For more information please check the documentation: [ML-Navigator](https://ki-labs.github.io/ML-Navigator/index.html)

## Modules

### flows

It is the core module of the ML-Navigator framework. It contains the flows that are defined as YAML files.
The Flow class has multiple methods that get their functionalities from calling other packages (self-implemented or external).

### preprocessing

It contains the main functions and classes that are used to prepare the data for further processing such as discovering
the type of data in each column, encode categorical features and scale numeric features.

## feature_engineering

It contains functions and classes to produce new features. For example, one-hot encoding.

## visualization

It contains functions and classes to produce graphs. For example, there are functions for comparing the statistical
 properties of different datasets, visualize the count of the missing values, and drawing histograms.

## training

It contains the functions and classes to train Machine Learning models. Currently, there are two regression models:
 Ridge linear regression (scikit-learn) and LightGBM.
 
### prediction

It contains the functions and classes to predict the target using the pre-trained models. Currently, all trained models are saved locally.

## Other directories

### logs

It contains the logs messages that are produced by different modules of the framework.

### models

It contains the trained models saved in pkl format.

### venv

The virtual environment that should be created by the user

### data

It contains the data, e.g. `train.csv` and `test.csv`.

## Files:

### requirements.txt

It contains a list of all packages that are required to run the framework.

## How to apply flows to your data

It is straightforward. You need to point to the location of your data and the name of the datasets after loading a particular flow.
Currently, the framework supports only reading CSV files. Here is an example:

````python
path = './data'
files_list = ['train.csv','test.csv']
````

## How to create a flow

### Introduction

For the first version, we have not prepared a tool for creating a flow in a cool way yet. However, this is one of the main focus in the near future.

The flow should contain two elements:

1. Visualization: I show a flow as a flowchart. I use a free online tool called draw.io to draw a chart. Feel free to use any other tool or method to visualize a flow.
You can use the `drawio` files which are provided in the flows modules to create a visualization of new flows.

2. guidance text: I use YAML files to define the guidance instructions for the flows. Currently, this method is not scalable, and it requires setting the steps manually.
In the future, the flows will be created using a user interface, and they will be saved in a database using a unique key for each flow.

Each method in the `Flows` class in the `flows/flow.py` has an ID. Each ID is defined as a string using the variable `function_id`. For example, the method `load_data` has `function_id = 0`.
In the `flows/flow_instructions_database.yaml` there are the instruction texts. The instruction texts are defined using an ID (integer). Each text has two variables:
* `function`: it describes the function or the method which is defined in `flows/flow.py`
* `guide`: it the guidance text that describe how to use the defined function or the method.

To build a flow you need to create a JSON file, e.g. `flow_0.json`, that map the `function_id` as a key and the ID of the guidance text that is defined in `flows/flow_instructions_database.yaml`.
<br>
<strong>IMPORTANT!!!!!</strong><br>
In the `flow_x.json`:<br>
`function_id` refers to the id of the current running function<br>
The ID of the guidance text in the `flow_instructions_database.yaml` refers to the function that should be executed after the current running function that has the `function_id`.

````json
{"function_id of running function": "the ID of the guidance text of the function that should be executed next"}
````

An example for the mapping:
````json
{"0": 1}
````

where `function_id = 0` refers to the `load_data` method and value 1 refers to the ID of the method `Encode categorical features` guidance text that should run after the `load_data` method.

### Example of creating the flow_0.json

````json
{
  "0": 1,
  "1": 2,
  "2": 3,
  "4": 1000
}
````

The translation of the JSON object is as follows:

````json
{
  "the current function: load the data": "the next function: Encode categorical features",
  "the current function: Encode categorical features": "the next function: Scale numeric features",
  "the current function: Scale numeric features": "the next function: Train a model",
  "the current function: Train a model": "the next function: Finish or noting which indicates the end of the flow"
}
````

### Creating new flows components

You can create your own method inside the `Flows` class in the `flows/flow.py` and assign a unique ID to it by defining the variable `function_id`.
Inside the `flow_instructions_database.yaml` you can create your own guidance text for already exiting methods or for the new methods. You should assign a unique ID for the 
new created guidance texts. Please include the `function` and `quide` keys to help other users understanding and find your guidance text easily.
The key `function` is optional but `quide` is required. You can create multiple new guidance texts for the same defined function but each guidance text should have a unique ID.

When creating a flow, the essential information that should be added at the end of each step is what the next step is. Adding an example, which shows how to perform
the next level and what are the required variables is beneficial to the user.

# Contribution

Your contributions are always welcome and appreciated. Following are the things you can do to contribute to this project.

## Report a bug
If you think you have encountered a bug, and I should know about it, feel free to report it [here](https://github.com/KI-labs/ML-Navigator/issues) and I will take care of it.

## Request a feature
You can also request for a feature [here](https://github.com/KI-labs/ML-Navigator/issues), and if it will viable, it will be picked for development.

## Create a pull request
It can't get better then this, your pull request will be appreciated by the community. You can get started by picking up any open issues from [here](https://github.com/KI-labs/ML-Navigator/issues) and make a pull request.

## create a new flow:
If you want to submit a flow, please provide the following in your pull request:
1. `flow_x.drawio` and `flow_x.png` where x is an integer that has not been given for other flows yet. Please check the flows module `./flows`<br>

2. `flow_x.json` where x is is an integer that has not been given for other flows yet and has the same value in `flow_x.drawio`<br>

3. `flow_x.ipynb` where x is is an integer that has not been given for other flows yet and has the same value in `flow_x.drawio` and in `flow_x.json`.<br>
The Jupyter Notebook `flow_x.ipynb` should work end-2-end without any errors.

Steps to create a pull request
------------------------------

1. Make a PR to master branch.
2. Comply with the best practices and guidelines.
3. It must pass all continuous integration checks and get positive reviews.
4. After this, changes will be merged.

# License

Copyright 2019 KI labs GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.