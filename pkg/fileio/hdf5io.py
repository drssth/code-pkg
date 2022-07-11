import os
import os.path
import h5py
import numpy as np
import pandas as pd

def get_keys(hdf5obj, key, layer):
    ls = []

    if key is None:
        ls = list(hdf5obj.keys())
    else:
        if isinstance(hdf5obj[key], h5py._hl.group.Group):
            ls = list(hdf5obj[key].keys())
        else:
            return [], layer+1

    if key is None:
        ls = ls
    else:
        ls = [os.path.join(key, i) for i in ls]

    return ls, layer+1



def get_hdf5_key(hdf5fn, layers=2):
    keys = []
    with h5py.File(hdf5fn, 'r') as hf:
        keys, layer_count = get_keys(hf, None, 0)

        while layer_count < layers:
            ls = []
            lc = layer_count
            for i in keys:
                lsi, lc = get_keys(hf, i, layer_count)
                ls.extend(lsi)
            keys, layer_count = ls, lc

    return keys


class Hdf5Interface(object):

    def __init__(self, filename):
        self.filename = filename
        self.keys = []
        self.get_keys()


    def save(self, key, data, overwrite=False):
        keyexists = False
        write_data = True
        if key in self.keys:
            keyexists = True

        if keyexists:
            if overwrite:
                write_data = True
                print('key exists already, overwrite data')
            else:
                write_data = False
                print('key exists already, skip')
        else:
            write_data = True

        if write_data:
            parent = os.path.dirname(self.filename)
            if len(parent) > 0:
                os.makedirs(parent, exist_ok=True)
            with h5py.File(self.filename, 'a') as hf:
                hf.create_dataset(key, data=data)
                self.keys.append(key)


    def load(self, key):
        data = None
        if os.path.exists(self.filename):
            with h5py.File(self.filename, 'r') as hf:
                if key in hf:
                    data = hf[key][:]
                else:
                    print('key does not exist: {}'.format(key))
        else:
            print('file does not exist: {}'.format(self.filename))
        return data


    def load_dataframe(self, key):
        df = pd.read_hdf(self.filename, key=key, mode='r')
        return df


    def get_keys(self, layers=1):
        assert isinstance(layers, int)
        if os.path.exists(self.filename):
            self.keys = get_hdf5_key(self.filename, layers=layers)
        return self.keys


    def key_exists(self, key):
        return key in self.keys()



def test_hdf5interface():
    data = np.random.random(size=(100,20))
    h = Hdf5Interface('testdata/data.h5')
    ls = h.get_keys()
    assert len(ls) == 1, 'success'
    h.save('d1', data)
    ls = h.get_keys()
    assert len(ls) == 2, 'success'
