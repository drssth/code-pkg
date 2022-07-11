import os
import os.path
import pandas as pd
import logging

def load_csv(filename, index_col=None):
    df = None
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=index_col)
        except:
            logging.error('cannot load csv {}'.format(filename))
    else:
        logging.error('file does not exist -- csv {}'.format(filename))
    return df


def save_csv(df, filename, index=False):
    parent = os.path.dirname(filename)
    if len(parent) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=index)
