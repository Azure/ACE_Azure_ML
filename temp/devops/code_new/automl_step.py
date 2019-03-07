
############################### load required libraries

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import urllib.request

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import azureml.core
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.telemetry import set_diagnostics_collection

############################### set up experiment

parser = argparse.ArgumentParser("automl_train")
parser.add_argument("--input_directory", default = "data", type = str, help = "input directory")
args = parser.parse_args()
print("input directory: %s" % args.input_directory)

run = Run.get_context()
ws = run.experiment.workspace

# Choose a name for the experiment and specify the project folder.
experiment_name = 'automl-local-classification'
project_folder = '.'
experiment = Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
print("Run info:", output)

set_diagnostics_collection(send_diagnostics = True)

############################### define functions

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


############################### get and preprocess data

df_telemetry, df_errors, df_fails, df_maint, df_machines = download_data()

df_left = df_telemetry.loc[:, ['datetime', 'machineID']] # we set this aside to this table to join all our results with

cols_to_average = df_telemetry.columns[-4:]

df_telemetry_rolling_3h = get_rolling_aggregates(df_telemetry, cols_to_average, 
                                                 suffixes = ['_ma_3', '_sd_3'], 
                                                 window = 3, on = 3, 
                                                 groupby = 'machineID', lagon = 'datetime')

df_telemetry_rolling_12h = get_rolling_aggregates(df_telemetry, cols_to_average, 
                                                  suffixes = ['_ma_12', '_sd_12'], 
                                                  window = 12, on = 3, 
                                                  groupby = 'machineID', lagon = 'datetime')

df_telemetry_rolling = pd.concat([df_telemetry_rolling_3h, df_telemetry_rolling_12h.drop(['machineID', 'datetime'], axis=1)], axis=1)

df_telemetry_feat_roll = df_left.merge(df_telemetry_rolling, how = "inner", on = ['machineID', 'datetime'], validate = "one_to_one")
df_telemetry_feat_roll.fillna(method = 'bfill', inplace = True)
# df_telemetry_feat_roll.head()

del df_telemetry_rolling, df_telemetry_rolling_3h, df_telemetry_rolling_12h
df_errors_feat_roll = get_datetime_diffs(df_left, df_errors, catvar = 'errorID', prefix = 'e', window = 6, lagon = 'datetime', on = 3)
# df_errors_feat_roll.tail()

df_errors_feat_roll.loc[df_errors_feat_roll['machineID'] == 2, :].head()

df_maint_feat_roll = get_datetime_diffs(df_left, df_maint, catvar = 'comp', prefix = 'm', 
                                        window = 6, lagon = 'datetime', on = 3, show_example = False)
# df_maint_feat_roll.tail()

df_maint_feat_roll.loc[df_maint_feat_roll['machineID'] == 2, :].head()

df_fails_feat_roll = get_datetime_diffs(df_left, df_fails, catvar = 'failure', prefix = 'f', 
                                        window = 6, lagon = 'datetime', on = 3, show_example = False)
# df_fails_feat_roll.tail()

assert(df_errors_feat_roll.shape[0] == df_fails_feat_roll.shape[0] == df_maint_feat_roll.shape[0] == df_telemetry_feat_roll.shape[0])

df_all = pd.concat([df_telemetry_feat_roll,
                    df_errors_feat_roll.drop(columns = ['machineID', 'datetime']), 
                    df_maint_feat_roll.drop(columns = ['machineID', 'datetime']), 
                    df_fails_feat_roll.drop(columns = ['machineID', 'datetime'])], axis = 1, verify_integrity = True)

df_all = pd.merge(left = df_all, right = df_machines, how = "left", on = 'machineID', validate = 'many_to_one')

del df_left, df_telemetry_feat_roll, df_errors_feat_roll, df_fails_feat_roll, df_maint_feat_roll

for i in range(1, 5): # iterate over the four components
    # find all the times a component failed for a given machine
    df_temp = df_all.loc[df_all['f_' + str(i)] == 1, ['machineID', 'datetime']]
    label = 'y_' + str(i) # name of target column (one per component)
    df_all[label] = 0
    for n in range(df_temp.shape[0]): # iterate over all the failure times
        machineID, datetime = df_temp.iloc[n, :]
        dt_end = datetime - pd.Timedelta('3 hours') # 3 hours prior to failure
        dt_start = datetime - pd.Timedelta('2 days') # n days prior to failure
        if n % 500 == 0: 
            print("a failure occured on machine {0} at {1}, so {2} is set to 1 between {4} and {3}".format(machineID, datetime, label, dt_end, dt_start))
        df_all.loc[(df_all['machineID'] == machineID) & 
                   (df_all['datetime'].between(dt_start, dt_end)), label] = 1

############################### run automl experiment

X_drop = ['datetime', 'machineID', 'f_1', 'f_2', 'f_3', 'f_4', 'y_1', 'y_2', 'y_3', 'y_4', 'model']
Y_keep = ['y_1', 'y_2', 'y_3', 'y_4']

X_train = df_all.loc[df_all['datetime'] < '2015-10-01', ].drop(X_drop, axis=1)
y_train = df_all.loc[df_all['datetime'] < '2015-10-01', Y_keep]

X_test = df_all.loc[df_all['datetime'] > '2015-10-15', ].drop(X_drop, axis=1)
y_test = df_all.loc[df_all['datetime'] > '2015-10-15', Y_keep]

primary_metric = 'AUC_weighted'

automl_config = AutoMLConfig(task = 'classification', 
                             preprocess = False,
                             name = experiment_name,
                             debug_log = 'automl_errors.log',
                             primary_metric = primary_metric,
                             max_time_sec = 1200,
                             iterations = 2,
                             n_cross_validations = 2,
                             verbosity = logging.INFO,
                             X = X_train.values, # we convert from pandas to numpy arrays using .vaules
                             y = y_train.values[:, 0], # we convert from pandas to numpy arrays using .vaules
                             path = project_folder, )

local_run = experiment.submit(automl_config, show_output = True)

# Wait until the run finishes.
local_run.wait_for_completion(show_output = True)

# create new AutoMLRun object to ensure everything is in order
ml_run = AutoMLRun(experiment = experiment, run_id = local_run.id)

# aux function for comparing performance of runs (quick workaround for automl's _get_max_min_comparator)
def maximize(x, y):
    if x >= y:
        return x
    else:
        return y

# next couple of lines are stripped down version of automl's get_output
children = list(ml_run.get_children())

best_run = None # will be child run with best performance
best_score = None # performance of that child run

for child in children:
    candidate_score = child.get_metrics()[primary_metric]
    if not np.isnan(candidate_score):
        if best_score is None:
            best_score = candidate_score
            best_run = child
        else:
            new_score = maximize(best_score, candidate_score)
            if new_score != best_score:
                best_score = new_score
                best_run = child    

# print accuracy                 
best_accuracy = best_run.get_metrics()['accuracy']
print("Best run accuracy:", best_accuracy)

# download model and save to pkl
model_path = "outputs/model.pkl"
best_run.download_file(name = model_path, output_file_path = model_path)

# Writing the run id to /aml_config/run_id.json
run_id = {}
run_id['run_id'] = best_run.id
run_id['experiment_name'] = best_run.experiment.name

# save run info 
os.makedirs('aml_config', exist_ok = True)
with open('aml_config/run_id.json', 'w') as outfile:
    json.dump(run_id, outfile)

############################### upload run info and model pkl to def_data_store

def_data_store = ws.get_default_datastore()
def_data_store.upload(src_dir = 'aml_config', target_path = 'aml_config', overwrite = True)
def_data_store.upload(src_dir = 'outputs', target_path = 'outputs', overwrite = True)
