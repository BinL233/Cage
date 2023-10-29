import json

def merge_parameters(parameters, default_parameters):
    with open(parameters, "r") as infile:
        parameters = json.load(infile)

    for parameter, value in default_parameters.items():
        if parameter not in parameters:
            if value is None and parameter != "controls":
                raise ValueError("Must provide value for '{}'".format(parameter))

            parameters[parameter] = value

    return parameters