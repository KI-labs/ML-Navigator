import yaml
from blessings import Terminal
from flow_instructions import FlowInstructions

term = Terminal()

with open("flow_instructions_database.yaml", 'r') as stream:
    try:
        commands = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
for command_key in commands.keys():
    for command in commands[command_key][0]["guide"]:
        eval(command)
