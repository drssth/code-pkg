import sys
import os
import os.path
import pandas as pd
import datetime
import logging



def format_time(df, timecol, savetimecol, timetype='str', format='%Y%m%d'):
    
    if ',' in timecol:
        cols = timecol.split(',')
        df_dt = df.get(cols)
        df_dt = df_dt.dropna()
        df = df.loc[df_dt.index,:]
        ns = df_dt.shape[0]
        
        if timetype == 'int':
            dt_list = [datetime.datetime(*df_dt.loc[i, :].tolist()) for i in df_dt.index]
            df[savetimecol] = dt_list
        elif timetype == 'str':
            dt_list = [' '.join(df_dt.loc[i, :].tolist()) for i in df_dt.index]
            df[savetimecol] = pd.to_datetime(dt_list, format=format)
        else:
            dt_list = [''.join(df_dt.loc[i, :].tolist()) for i in df_dt.index]
            df[savetimecol] = pd.to_datetime(dt_list, format=format)
            
    else:
        df_dt = df.get([timecol])
        df_dt = df_dt.dropna()
        df = df.loc[df_dt.index,:]
        df[savetimecol] = pd.to_datetime(df[timecol], format=format)
        
    return df



def reindex_time_series(df, freq):
    start_day, end_day = min(df.index), max(df.index)
    timeindex = pd.date_range(start=start_day, end=end_day, freq=freq)
    df = df.reindex(timeindex)
    return df

    

def get_time_index(time_year_span, freq='H'):
    sy, ey = time_year_span
    time_index = pd.date_range('{:04}-01-01'.format(sy), '{:04}-01-01'.format(ey), freq=freq)
    time_index = time_index[0:-1]
    return time_index


def is_excluded_days(t, exclude_days):
    y, m, d = t.year, t.month, t.day
    for yy, mm, dd in exclude_days:
        if yy == '*':
            if (m == mm) and (d == dd):
                return True
        else:
            if (y == yy) and (m == mm) and (d == dd):
                return True
    return False


def generate_time_index(time_year_span, freq='H', exclude_days=[('*', 2, 29)]):
    time_index = get_time_index(time_year_span, freq='H')
    time_list = [i for i in time_index if not is_excluded_days(i, exclude_days)]
    return time_list


def test_generate_time_index():
    ls = generate_time_index((1995, 2006), exclude_days=[
        ('*', 2, 29), (2001, 1, 23), (2003, 3, 16)])
    assert len(ls) == 96312, 'success'

    ls = generate_time_index((1995, 2006), exclude_days=[
        ('*', 2, 29),])
    assert len(ls) == 96360, 'success'
