import os

directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(directory):
    os.makedirs(directory)
