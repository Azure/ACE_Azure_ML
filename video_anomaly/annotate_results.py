from settings import RESULTS_DIR, DATA_DIR
import pandas as pd
import os

nt = 200 # length of video sequences
dataset = 'UCSDped1'

# set path for saving output
save_path = os.path.join(RESULTS_DIR, dataset)

# Load the result of fitting the trained model to the test data
df = pd.read_pickle(os.path.join(save_path, 'test_results.pkl.gz'))

# The next step is to merge that with the labels for anomalies in the ucsd dataset 
# UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/UCSDped1.m
anom_anot_filename = os.path.join(DATA_DIR, dataset, 'Test/%s.m' % dataset)
with open(anom_anot_filename, 'r') as f:
    lines = f.readlines()
del lines[0]

# extract the beginning and end of subsequences that contain anomalies
anom_indices = []
for l, line in enumerate(lines):
    line = line.replace(":", ",")
    anom_indices.append(line.split("[")[1].split("]")[0].split(","))

# add column with anomalies to df
df['anom'] = 0
for a, anom in enumerate(anom_indices):
    while len(anom) > 0:
        first_frame = int(anom.pop(0)) + a * nt
        last_frame = int(anom.pop(0)) + a * nt
        print(first_frame, last_frame)
        df.loc[first_frame:last_frame, 'anom'] = 1

# save the dataframe 
df.to_pickle(os.path.join(save_path, 'df.pkl.gz'))

# # For confirmation, you can create a plot that should show blocks of anomalies
# import matplotlib.pyplot as plt
# plt.plot(df['anom'])
# plt.show()

