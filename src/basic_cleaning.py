import os
import pandas as pd
import yaml

# Read metadata file
metadata = yaml.safe_load(open('metadata.yaml'))


def __clean_dataset(df):
    """
    Clean the dataset by dropping null values and specified columns.
    """
    # Drop rows with null values
    df.dropna(inplace=True)

    # Drop specified columns from metadata file
    for col in metadata['columns_to_drop']:
        df.drop(col, axis="columns", inplace=True)
    return df


def execute_cleaning():
    """
    Execute data cleaning by reading the raw data,
    cleaning it, and saving it to a new CSV file.
    """
    # Read raw data from CSV file
    df = pd.read_csv(
        "data/raw/census.csv",
        skipinitialspace=True,
        na_values='?')

    # Clean the dataset
    df = __clean_dataset(df)

    # Create directory if it doesn't exist
    os.makedirs("data/prepared", exist_ok=True)

    # Save cleaned dataset to new CSV file
    df.to_csv("data/prepared/census.csv", index=False)


if __name__ == "__main__":
    execute_cleaning()
