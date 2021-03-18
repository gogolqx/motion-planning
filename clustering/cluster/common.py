"""Modul for general functions."""
import pandas as pd
import csv


def read_csv(file, delimiter='\t'):
    """Read csv with specified delimiter."""
    return pd.read_csv(file, delimiter=delimiter, index_col=0)


def rename(df):
    """Rename dataframe column."""
    df = df.rename(columns={'scenario_id': 'scenario_name'})
    df['scenario_id'] = df.index
    return df


def dict_to_csv(dict, path):
    """Convert dictionary to csv file."""
    w = csv.writer(open(path, "w"))
    for key, val in dict.items():
        w.writerow([key, val])
