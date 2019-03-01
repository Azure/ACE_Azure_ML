import pandas as pd
import numpy as np
import urllib.request
import os

def download_data():
    os.makedirs('../data', exist_ok = True)
    container = 'https://sethmottstore.blob.core.windows.net/predmaint/'
    urllib.request.urlretrieve(container + 'telemetry.csv', filename='../data/telemetry.csv')
    urllib.request.urlretrieve(container + 'maintenance.csv', filename='../data/maintenance.csv')
    urllib.request.urlretrieve(container + 'machines.csv', filename='../data/machines.csv')
    urllib.request.urlretrieve(container + 'failures.csv', filename='../data/failures.csv')
    # we replace errors.csv with anoms.csv (results from running anomaly detection)
    # urllib.request.urlretrieve(container + 'errors.csv', filename='../data/errors.csv')
    urllib.request.urlretrieve(container + 'anoms.csv', filename='../data/anoms.csv')
    df_telemetry = pd.read_csv('../data/telemetry.csv', header=0)
    df_telemetry['datetime'] = pd.to_datetime(df_telemetry['datetime'], format="%m/%d/%Y %I:%M:%S %p")
    df_errors = pd.read_csv('../data/anoms.csv', header=0)
    df_errors['datetime'] = pd.to_datetime(df_errors['datetime'])
    rep_dir = {"volt":"error1", "rotate":"error2", "pressure":"error3", "vibration":"error4"}
    df_errors = df_errors.replace({"errorID": rep_dir})
    df_subset = df_errors.loc[(df_errors.datetime.between('2015-01-01', '2016-01-01')) & (df_errors.machineID == 1)]
    df_subset.head()
    df_fails = pd.read_csv('../data/failures.csv', header=0)
    df_fails['datetime'] = pd.to_datetime(df_fails['datetime'], format="%m/%d/%Y %I:%M:%S %p")
    df_maint = pd.read_csv('../data/maintenance.csv', header=0)
    df_maint['datetime'] = pd.to_datetime(df_maint['datetime'], format="%m/%d/%Y %I:%M:%S %p")
    df_machines = pd.read_csv('../data/machines.csv', header=0)
    df_errors['errorID'] = df_errors['errorID'].apply(lambda x: int(x[-1]))
    df_maint['comp'] = df_maint['comp'].apply(lambda x: int(x[-1]))
    df_fails['failure'] = df_fails['failure'].apply(lambda x: int(x[-1]))
    return df_telemetry, df_errors, df_subset, df_fails, df_maint, df_machines

def get_datetime_diffs(df_left, df_right, catvar, prefix, window, on, lagon = None, diff_type = 'timedelta64[h]', validate = 'one_to_one', show_example = True):
    keys = ['machineID', 'datetime']
    df_dummies = pd.get_dummies(df_right[catvar], prefix=prefix)
    df_wide = pd.concat([df_right.loc[:, keys], df_dummies], axis=1)
    df_wide = df_wide.groupby(keys).sum().reset_index()
    df = df_left.merge(df_wide, how="left", on=keys, validate = validate).fillna(0)
    # run a rolling window through event flags to aggregate data
    dummy_col_names = df_dummies.columns
    df = df.groupby('machineID').rolling(window=window, on=lagon)[dummy_col_names].max()
    df.reset_index(inplace=True)
    df = df.loc[df.index % on == on-1]
    df.reset_index(inplace=True, drop=True)
    df_first = df.groupby('machineID', as_index=False).nth(0)
    # calculate the time of the last event and the time elapsed since
    for col in dummy_col_names:
        whenlast, diffcol = 'last_' + col, 'd' + col
        df.loc[:, col].fillna(value = 0, inplace=True)
        # let's assume an event happened in row 0, so we don't have missing values for the time elapsed
        df.iloc[df_first.index, df.columns.get_loc(col)] = 1
        df.loc[df[col] == 1, whenlast] = df.loc[df[col] == 1, 'datetime']
        # for the first occurence we don't know when it last happened, so we assume it happened then
        df.iloc[df_first.index, df.columns.get_loc(whenlast)] = df.iloc[df_first.index, df.columns.get_loc('datetime')]
        df[whenlast].fillna(method='ffill', inplace=True)
        # df.loc[df[whenlast] > df['datetime'], whenlast] = np.nan
        df.loc[df[whenlast] <= df['datetime'], diffcol] = (df['datetime'] - df[whenlast]).astype(diff_type)
        df.drop(columns = whenlast, inplace=True)
    if show_example == True:
        col = np.random.choice(dummy_col_names, size = 1)[0]
        idx = np.random.choice(df.loc[df[col] == 1, :].index.tolist(), size = 1)[0]
        print('Example:\n')
        print(df.loc[df.index.isin(range(idx-3, idx+5)), ['datetime', col, 'd' + col]])    
    return df

def get_rolling_aggregates(df, colnames, suffixes, window, on, groupby, lagon = None):
    """
    calculates rolling averages and standard deviations
    
    Arguments:
    df -- dataframe to run it on
    colnames -- names of columns we want rolling statistics for
    suffixes -- suffixes attached to the new columns (provide a list with strings)
    window -- the lag over which rolling statistics are calculated
    on -- the interval at which rolling statistics are calculated
    groupby -- the column used to group results by
    lagon -- the name of the datetime column used to compute lags (if none specified it defaults to row number)
      
    Returns:
    a dataframe with rolling statistics over a specified lag calculated over a specified interval
    """
    
    rolling_colnames = [c + suffixes[0] for c in colnames]
    df_rolling_mean = df.groupby(groupby).rolling(window=window, on=lagon)[colnames].mean()
    df_rolling_mean.columns = rolling_colnames
    df_rolling_mean.reset_index(inplace=True)
    
    rolling_colnames = [c + suffixes[1] for c in colnames]
    df_rolling_sd = df.groupby(groupby).rolling(window=window, on=lagon)[colnames].var()
    df_rolling_sd.columns = rolling_colnames
    df_rolling_sd = df_rolling_sd.apply(np.sqrt)
    df_rolling_sd.reset_index(inplace=True, drop=True)
    
    df_res = pd.concat([df_rolling_mean, df_rolling_sd], axis=1)
    df_res = df_res.loc[df_res.index % on == on-1]
    return df_res
 
