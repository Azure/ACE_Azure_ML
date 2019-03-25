import unittest
import sys
import os
import pandas as pd
import pytest

data = pd.DataFrame()
n_columns = 37

@pytest.fixture()
def get_sample(name):
    global data
    filename = name
    data = pd.read_csv(filename)

def test_schema_types(get_sample):
    for col in data:
        try:
            float(col)
        except ValueError:
            raise AssertionError


def test_schema_cols(get_sample):
    n_actual_columns = data.shape[1]
    assert n_actual_columns == n_columns
