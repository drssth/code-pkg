import os
import os.path
import numpy
import logging


def load_npz(filename):
    data = None
    if os.path.exists(filename):
        try:
            data = numpy.load(filename)
        except:
            logging.error('cannot load numpy data {}'.format(filename))
    else:
        logging.error('file does not exist -- numpy data {}'.format(filename))

    if data is not None:
        if 'data' in data:
            data = data['data']
    return data


def save_npz(data, filename):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    numpy.savez_compressed(filename, data=data)
