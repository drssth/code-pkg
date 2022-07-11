import os
import os.path
import json
import logging


def load_json(filename):
    data = None
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except:
            logging.error('cannot load json {}'.format(filename))
    else:
        logging.error('file does not exist -- json {}'.format(filename))
    return data


def save_json(data, filename):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True, indent='   ')


def pretty_print_json(data):
    return json.dumps(data, indent=4, sort_keys=True)