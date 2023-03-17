"""
Basic cleaning procedure
"""
import pandas as pd
import yaml

# Read metadata file
metadata = yaml.safe_load(open('../metadata.yaml'))


def __clean_dataset(df):
    """
    Clean the dataset doing some stuff got from eda
    """
    df.dropna(inplace=True)
    for col in metadata['columns_to_drop']:
        df.drop(col, axis="columns", inplace=True)
    return df


def execute_cleaning():
    """
    Execute data cleaning
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True, na_values='?')
    df = __clean_dataset(df)
    df.to_csv("data/prepared/census.csv", index=False)
