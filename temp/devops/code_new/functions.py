
def download_data():
    """
    download the anomaly detection and predictive maintenance data
    :return: all the data
    """
    os.makedirs('../data', exist_ok = True)
    container = 'https://coursematerial.blob.core.windows.net/data/telemetry/'
    
    urllib.request.urlretrieve(container + 'telemetry.csv', filename = '../data/telemetry.csv')
    urllib.request.urlretrieve(container + 'maintenance.csv', filename = '../data/maintenance.csv')
    urllib.request.urlretrieve(container + 'machines.csv', filename = '../data/machines.csv')
    urllib.request.urlretrieve(container + 'failures.csv', filename = '../data/failures.csv')
    # we replace errors.csv with anoms.csv (results from running anomaly detection)
    # urllib.request.urlretrieve(container + 'errors.csv', filename = '../data/errors.csv')
    urllib.request.urlretrieve(container + 'anoms.csv', filename = '../data/anoms.csv')
    
    df_telemetry = pd.read_csv('../data/telemetry.csv', header = 0)
    df_errors = pd.read_csv('../data/anoms.csv', header = 0)
    df_fails = pd.read_csv('../data/failures.csv', header = 0)
    df_maint = pd.read_csv('../data/maintenance.csv', header = 0)
    df_machines = pd.read_csv('../data/machines.csv', header = 0)

    df_telemetry['datetime'] = pd.to_datetime(df_telemetry['datetime'], format = "%m/%d/%Y %I:%M:%S %p")

    df_errors['datetime'] = pd.to_datetime(df_errors['datetime'])
    rep_dir = {"volt":"error1", "rotate":"error2", "pressure":"error3", "vibration":"error4"}
    df_errors = df_errors.replace({"errorID": rep_dir})
    df_errors['errorID'] = df_errors['errorID'].apply(lambda x: int(x[-1]))

    df_fails['datetime'] = pd.to_datetime(df_fails['datetime'], format = "%m/%d/%Y %I:%M:%S %p")
    df_fails['failure'] = df_fails['failure'].apply(lambda x: int(x[-1]))

    df_maint['datetime'] = pd.to_datetime(df_maint['datetime'], format = "%m/%d/%Y %I:%M:%S %p")
    df_maint['comp'] = df_maint['comp'].apply(lambda x: int(x[-1]))
    
    return df_telemetry, df_errors, df_fails, df_maint, df_machines


def get_rolling_aggregates(df, colnames, suffixes, window, on, groupby, lagon = None):
    """
    calculates rolling averages and standard deviations
    
    :param df: dataframe to run it on
    :param colnames: names of columns we want rolling statistics for
    :param suffixes: suffixes attached to the new columns (provide a list with strings)
    :param window: the lag over which rolling statistics are calculated
    :param on: the interval at which rolling statistics are calculated
    :param groupby: the column used to group results by
    :param lagon: the name of the datetime column used to compute lags (if none specified it defaults to row number)
    :return: a dataframe with rolling statistics over a specified lag calculated over a specified interval
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


def get_datetime_diffs(df_left, df_right, catvar, prefix, window, on, lagon = None, diff_type = 'timedelta64[h]', validate = 'one_to_one', show_example = True):
    """
    calculates for every timestamp the time elapsed since the last time an event occured where an event is either an error or anomaly, maintenance, or failure
    
    :param df_left: the telemetry data collected at regular intervals
    :param df_right: the event data collected at irregular intervals
    :param catvar: the name of the categorical column that encodes the event
    :param prefix: the prefix for the new column showing time elapsed
    :param window: window size for detecting event
    :param on: frequency we want the results to be in
    :param lagon: the name of the datetime column used to compute lags (if none specified it defaults to row number)
    :param diff_type: the unit we want time differences to be measured in (hour by default)
    :param validate: whether we should validate results
    :param show_example: whether we should show an example to check that things are working
    :return: a dataframe with rolling statistics over a specified lag calculated over a specified interval
    """

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


def rolling_average(df, column, n = 24):
    """
    Calculates rolling average according to Welford's online algorithm (Donald Knuth's Art of Computer Programming, Vol 2, page 232, 3rd edition).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    
    This adds a column next to the column of interest, with the suffix '_<n>' on the column name

    :param df: a dataframe with time series in columns
    :param column: name of the column of interest
    :param n: number of measurements to consider
    :return: None
    """

    ra = [0] * df.shape[0]
    ra[0] = df[column].values[0]

    for r in range(1, df.shape[0]):
        curr_n = float(min(n, r))
        ra[r] = ra[r-1] + (df[column].values[r] - ra[r-1])/curr_n

    df = pd.DataFrame(data = {'datetime': df['datetime'], 'value': ra})

    return df


def do_ad(df, alpha = 0.005, max_anoms = 0.1, only_last = None, longterm = False, e_value = False, direction = 'both'):
    """
    This method performs the actual anomaly detection.  Expecting the a dataframe with multiple sensors,
    and a specification of which sensor to use for anomaly detection.

    :param df: a dataframe with a timestamp column and one more columns with telemetry data
    :param column: name of the column on which to perform AD
    :param alpha: see pyculiarity documentation for the meaning of these parameters
    :param max_anoms:
    :param only_last:
    :param longterm:
    :param e_value:
    :param direction:
    :return: a pd.Series containing anomalies.  If not an anomaly, entry will be NaN, otherwise the sensor reading
    """
    results = detect_ts(df,
                        max_anoms = max_anoms,
                        alpha = alpha,
                        direction = direction,
                        e_value = e_value,
                        longterm = longterm,
                        only_last = only_last)

    return results['anoms']['timestamp'].values
