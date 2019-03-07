# test integrity of the input data 

import sys
import os
import numpy as np
import pandas as pd

# number of features
n_columns = 37
def check_schema(X):
    n_actual_columns = X.shape[1]
    if n_actual_columns != n_columns:
        print("Error: found {} feature columns. The data should have {} feature columns.".format(n_actual_columns, n_columns))
        return False
    return True

def main():
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print("Error: The file {} does not exist".format(filename))
        return

    dataset = pd.read_csv(filename)
    if check_schema(dataset[dataset.columns[:-1]]): 
        print("Data schema test succeeded")
    else:
        print("Data schema test failed")
if __name__ == "__main__":
    main()


 
