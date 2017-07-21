"""
    This file hold the functionality to determine the
    data quality of a given dataset by implementing
    a data score using a tanh(1/x) reward for duplicated
    values where x corresponds the number of duplicates.
    Additionally, falisfied data values will be punished
    by subtracting the distance weighted over the former value
    for a numerical attribute.
"""
from functools import partial
import multiprocessing
import pandas as pd
import numpy as np
import math

def compare_series(new_series,original_series):
    """
    Comparing two given series and determining the offset between those
    """
    new_series_numeric = new_series.apply(lambda x: float(x) if str(x).isnumeric() else 0)
    original_series_numeric = original_series.apply(lambda x: float(x) if str(x).isnumeric() else 0)
    return original_series_numeric.subtract(new_series_numeric)


def reward_unique_values(number_of_unique_entries,total_entries):
    """
    Rewarding the number of unique values over the total number
    """
    #return (number_of_unique_entries/total_entries*100)
    return number_of_unique_entries

def reward_duplicated_values(value_counts):
    """
    For each duplicate, we add to the score tanh(1/x) as reward
    where x is the number of duplicates exist
    """
    partial_score = 0
    for value,count in value_counts.iteritems():
        if count == 1:
            continue
        number_of_additional_duplicates = count-1
        partial_score += np.tanh(1/number_of_additional_duplicates)
    return partial_score

def punish_falsified_values(new_series,original_series):
    """
    Pushing falisfied data values by comparing their distance weighted
    over the original data value.
    This method may only be used for numerical series (attributes)
    """
    difference_series = compare_series(new_series,original_series)
    if difference_series is None:
        return 0
    negative_score = 0
    for index,difference in difference_series.iteritems():
        original_value = original_series[index]
        if not str(difference).isdigit() or not str(original_value).isdigit():
            continue
        negative_score += difference/original_value
    return negative_score

def evaluate_column(df_local,original_df=False):
    """
    Calculate the score for a pandas series / DF column if no comparison ground is available.
    This method serves all cases when actual values are not falsified
    """
    number_of_unique_entries = len(df_local["treated"].unique())
    total_entries = len(df_local["treated"].index)
    value_counts = df_local["treated"].value_counts(sort=True,dropna=True) # sort by values & exclude NaNs

    score = 0
    score += reward_unique_values(number_of_unique_entries,total_entries)
    score += reward_duplicated_values(value_counts)
    if original_df:
        score -= punish_falsified_values(df_local["treated"],df_local["original"])

    return score

def estimate_data_quality(df,colnames,num_cores,original_df=None):
    """
    Given a dataframe, iterating over all columns as pandas series and separately
    evaluating there score through multiprocessing.
    Finally sum up all values.
    """
    number_of_attributes = len(colnames)
    score = 0

    if original_df is None:
        with multiprocessing.Pool(num_cores) as pool:
            for i in pool.imap_unordered(partial(evaluate_column,original_df=False), [df[colname].to_frame(name="treated") for colname in colnames]):
                if i is not None:
                    score += i
            pool.close()
            pool.join()
    else:
        with multiprocessing.Pool(num_cores) as pool:
            for i in pool.imap_unordered(partial(evaluate_column,original_df=True), [pd.concat([df[colname], original_df[colname]], keys=["treated","original"], axis=1) for colname in colnames]):
                if i is not None:
                    score += i
            pool.close()
            pool.join()


    return (score * number_of_attributes)
