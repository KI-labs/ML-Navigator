How-to Guides
=============

How to apply flows to your data
-------------------------------

It is straightforward. You need to point to the location of your data and the name of the datasets after loading a particular flow.
Currently, the framework supports only reading CSV files. Here is an example:

.. code:: python

    path = './data'
    files_list = ['train.csv','test.csv']

How to create a flow
--------------------

Introduction
~~~~~~~~~~~~

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

**IMPORTANT!!!!!**

In the `flow_x.json`:
`function_id` refers to the id of the current running function
The ID of the guidance text in the `flow_instructions_database.yaml` refers to the function that should be executed after the current running function that has the `function_id`.

.. code:: json

    {"function_id of running function": "the ID of the guidance text of the function that should be executed next"}

An example for the mapping:

.. code:: json

    {"0": 1}


where `function_id = 0` refers to the `load_data` method and value 1 refers to the ID of the method `Encode categorical features` guidance text that should run after the `load_data` method.

Example of creating the flow_0.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: json

    {
      "0": 1,
      "1": 2,
      "2": 3,
      "4": 1000
    }


The translation of the JSON object is as follows:

.. code:: json

    {
      "the current function: load the data": "the next function: Encode categorical features",
      "the current function: Encode categorical features": "the next function: Scale numeric features",
      "the current function: Scale numeric features": "the next function: Train a model",
      "the current function: Train a model": "the next function: Finish or noting which indicates the end of the flow"
    }


Creating new flows components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create your own method inside the `Flows` class in the `flows/flow.py` and assign a unique ID to it by defining the variable `function_id`.
Inside the `flow_instructions_database.yaml` you can create your own guidance text for already exiting methods or for the new methods. You should assign a unique ID for the
new created guidance texts. Please include the `function` and `quide` keys to help other users understanding and find your guidance text easily.
The key `function` is optional but `quide` is required. You can create multiple new guidance texts for the same defined function but each guidance text should have a unique ID.

When creating a flow, the essential information that should be added at the end of each step is what the next step is. Adding an example, which shows how to perform
the next level and what are the required variables is beneficial to the user.

If you want to submit a flow, please provide the following in your pull request:

1. `flow_x.drawio` where x is an integer that has not been given for other flows yet. Please check the flows module `./flows`

2. `flow_x.yaml` where x is is an integer that has not been given for other flows yet and has the same value in `flow_x.drawio`

3. `flow_x.ipynb` where x is is an integer that has not been given for other flows yet and has the same value in `flow_x.drawio` and in `flow_x.yaml`.

The Jupyter Notebook `flow_x.ipynb` should work end-2-end without any errors.
