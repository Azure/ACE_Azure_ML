import argparse
import pickle
import pandas as pd
import os
from pyculiarity import detect_ts # python port of Twitter AD lib
from pytictoc import TicToc # so we can time our operations

def rolling_average(df, column, n=24):
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

    df = pd.DataFrame(data={'datetime': df['datetime'], 'value': ra})
    return df


def do_ad(df, alpha=0.005, max_anoms=0.1, only_last=None, longterm=False, e_value=False, direction='both'):
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
                        max_anoms=max_anoms,
                        alpha=alpha,
                        direction=direction,
                        e_value=e_value,
                        longterm=longterm,
                        only_last=only_last)

    return results['anoms']['timestamp'].values


parser = argparse.ArgumentParser("anom_detect")

parser.add_argument("--output_directory", type=str, help="output directory")
args = parser.parse_args()

print("output directory: %s" % args.output_directory)
os.makedirs(args.output_directory, exist_ok=True)

# public store of telemetry data
data_dir = 'https://dataandaiworkshop.blob.core.windows.net/bootcamp/telemetry/'

print("Reading data ... ", end="")
telemetry = pd.read_csv(os.path.join(data_dir, 'telemetry.csv'))
print("Done.")

print("Adding incremental data...")
telemetry_incremental = pd.read_csv(os.path.join('devops/data_sample/', 'telemetry_incremental.csv'))
telemetry = telemetry.append(telemetry_incremental, ignore_index=True)
print("Done.")

print("Parsing datetime...", end="")
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%m/%d/%Y %I:%M:%S %p")
print("Done.")

window_size = 12  # how many measures to include in rolling average
sensors = telemetry.columns[2:] # sensors are stored in column 2 on
window_sizes = [window_size] * len(sensors)  # this can be changed to have individual window_sizes for each sensor
machine_ids = telemetry['machineID'].unique()

t = TicToc()
for machine_id in machine_ids[:1]: # TODO: make sure to remove the [:2], this is just here to allow us to test this
    df = telemetry.loc[telemetry.loc[:, 'machineID'] == machine_id, :]
    t.tic()
    print("Working on sensor: ")
    for s, sensor in enumerate(sensors):
        N = window_sizes[s]
        print("   %s " % sensor)
        
        df_ra = rolling_average(df, sensor, N)
        anoms_timestamps = do_ad(df_ra)
        
        df_anoms = pd.DataFrame(data={'datetime': anoms_timestamps, 'machineID': [machine_id] * len(anoms_timestamps), 'errorID': [sensor] * len(anoms_timestamps)})
        
        # if this is the first machine and sensor, we initialize a new dataframe
        if machine_id == 1 and s == 0:
            df_anoms_all = df_anoms
        else: # otherwise we append the newly detected anomalies to the existing dataframe
            df_anoms_all = df_anoms_all.append(df_anoms, ignore_index=True)

    # store of output
    obj = {}
    obj["df_anoms"] = df_anoms_all

    out_file = os.path.join(args.output_directory, "anoms.pkl")
    with open(out_file, "wb") as fp:
        pickle.dump(obj, fp)
        
    t.toc("Processing machine %s took" % machine_id)
