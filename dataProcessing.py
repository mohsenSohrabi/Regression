import pandas as pd
import numpy as np

def read_data(path, header):
    """

    :param path: the path for the dataset
    :param header: the header for the dataset

    """

    df = pd.read_fwf(path, na_values="?")
    df.columns = header
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def clean_data(df):

    """
    :param df: the input dataframe we want to clean
    auto_mpg has some missing values that we are going to fill them by 0
    Also it the last column ('car name') has values that are unique for all records
    so it is not useful for regression and we can just drop it
    """
    # replace null values with 0
    df.fillna(0)
    # drop the last column
    df.drop('car name', axis=1, inplace=True)
    return df

def get_X_Y_from_dataframe(df):
    """
    According to the website I got data from, the column with name "mpg" must be predicted
    so we get it as Y and then delete it from dataframe an the remaining will be X
    :param df: dataframe we work on
    :return: X and Y
    """
    Y = df['mpg'].values

    df.drop('mpg', axis=1, inplace=True)
    X = df.values

    # Because of working on nan values it is necessary to convert nan values to numbers
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)

    return X, Y




