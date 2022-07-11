import sys, os, os.path
import numpy as np
import pandas as pd

def remove_extra_cols(df, removecols, targetcol, additionalcols=[]):
    removecols = [i for i in removecols if len(i) > 0]
    
    extra_cols = [targetcol]
    if isinstance(additionalcols, list):
        extra_cols = [targetcol] + additionalcols
    
    
    cols = list(df)
    for i in extra_cols+removecols:
        if i in cols:
            cols.remove(i)

    cols.sort()
    df = df.get(extra_cols+cols)
    return df



def add_last_measurement(df, targetcol, lastmcol, freq='D'):
    ns, nc = df.shape
    lastday_m = []
    if freq == 'D':
        lastday_m = [np.nan] + df[targetcol].tolist()[0:ns-1]
    elif freq == 'H':
        lastday_m = [np.nan] * 24 + df[targetcol].tolist()[0:ns-24]
    df[lastmcol] = lastday_m
    return df



def add_next_measurement(df, targetcol, nextmcol, freq='D'):
    ns, nc = df.shape
    nextd_m = []
    if freq == 'D':
        nextd_m = df[targetcol].tolist()[1:] + [np.nan]
    elif freq == 'H':
        nextd_m = df[targetcol].tolist()[24:] + [np.nan] * 24
    df[nextmcol] = nextd_m
    return df