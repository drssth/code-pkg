import os
import os.path
import pickle
import logging


def load_pkl(filename):
    data = None
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except:
            logging.error('cannot load pickle {}'.format(filename))
    else:
        logging.error('file does not exist -- pickle {}'.format(filename))
    return data


def save_pkl(data, filename):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
