"""
    This file holds the major functionality to anonymize datasets.
    It implements the optimized exp. and W[2]-completed search of
    quasi-identifier (2nd class identifier / mpmUCCs) as identification
    step and also several treatment methods including local suppression,
    global generalization, perturbation and compartmentation.
"""

import numpy
import pandas as pd
import os
import sys
import time
import multiprocessing
from contextlib import closing
from functools import partial
import scipy
import scipy.stats
import itertools, collections
import logging
import logging.handlers
import operator as op
import functools
import argparse
import math
import random
import networkx
import pickle
from pathlib import Path
# own
#from clustering import *
from analyzer import *
from evaluation import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('./')
#sys.path.append('../')

parser = argparse.ArgumentParser()

parser.add_argument("-ncol", "--columns", default = 12, help="Amount of columns to be considered", type=int)
parser.add_argument("-exact", "--exact", default = 'no', help="Determining 2nd class identifier exact instead of heuristic", type=str)
parser.add_argument("-treat", "--treatment", default = 'yes', help="Shall treatment occur?", type=str)
parser.add_argument("-evaltreat", "--eval_treatment", default = 'yes', help="Shall treatment be evaluated?", type=str)
parser.add_argument("-treatment", "--overwrite_treatment", choices=['no',"suppression","generalization","compartmentation","perturbation"], default = 'no', help="Overwrite Treatment?", type=str)
parser.add_argument("-analyze", "--analyze_results", default = 'no', help="Shall quasi identifier be analyzed?", type=str)
parser.add_argument("-score", "--data_value_score", default = 'no', help="Shall data value be evaluated?", type=str)
parser.add_argument("-size", "--data_size", default = 'no', help="Shall data size be logged?", type=str)
parser.add_argument("-visualize", "--visualize", default = 'no', help="Shall visualize be created?", type=str)
parser.add_argument("-cache", "--cache", default = 'yes', help="Shall mpmUCCs be cached?", type=str)

args = parser.parse_args()

settings = {
    'source_data': 'extended_user_data_V2.csv.gz',
    #'source_data':'MERGED_HEALTH_CLOUD_DATA.large.csv.gz',
    'data_path': '../../',
    'max_rows': 1000000,#500000, # set 0 for unlimited
    'upper_threshold_supression': 100, # in percent
    'upper_threshold_perturbation': 50, # in percent
    'upper_threshold_generalization': 100, # in percent
    'threshold_id_column': 80, # in percent
    'threshold_cardinality': 30,
    'num_cores': (multiprocessing.cpu_count()),
    'amount_of_columns': args.columns,
    'max_chunksize_per_processor': 10000,
    'find_2nd_class_identifier_exact': args.exact,
    'do_treatment': args.treatment,
    'eval_treatment': args.eval_treatment,
    'overwrite_treatment': args.overwrite_treatment,
    'analyze_results': args.analyze_results,
    'measure_data_score': args.data_value_score,
    'measure_data_size': args.data_size,
    'create_visualize': args.visualize,
    'cache_2nd_class_identifiers': args.cache,
    'min_weight_to_be_considered': 7,
    'overall_min_mean_weight': 5,
    'min_mean_weight': 7,
    'min_len_for_min_mean_weight': 4,
    'musts_contain_CRM': True
}


#################################### helper ########################################################


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def measure_weight(column_combination):
    """
    Computes the sum of the weights for a provided column combination based on
    the previously calculated cardinality of each individual column.
    Therefore, the global variable colnames_weight will be accessed
    as well as colname_id_mapping for resolving the ID mapping.
    """
    return math.ceil(sum(colnames_weight[colname_id_mapping[x]] for x in column_combination))

def get_filename(data):
    if not settings['overwrite_treatment'] == "no":
        return "output/"+settings['overwrite_treatment']+"_"+data+".txt"
    else:
        return "output/"+data+".txt"

def measure_mean_weight(column_combination):
    """
    Computes the mean of all weights for a provided column combination based on
    the previously calculated cardinality of each individual column.
    Therefore, the global variable colnames_weight will be accessed
    as well as colname_id_mapping for resolving the ID mapping.
    """
    column_selection = [colnames_weight[colname_id_mapping[x]] for x in column_combination]
    if not column_selection:
        return 0
    estimated_mean = numpy.mean(column_selection)
    return math.ceil(estimated_mean)


def eval_colname_for_filtering(combination):
    """
    Evaluate whether a given combination shall be filtered or not depending whether
    its a superset of an already fined idenitifier or matches certain settings like
    thresholds or includes a column marked as CRM
    """
    contains_crm_colname = False

    for quasi_identifier in quasi_identifier_combinations:
        if set(combination).issuperset(quasi_identifier):
            return

    if settings['musts_contain_CRM']:
        for colname in resolve_id_mapping_combinations(combination):
            if colname in crm_colnames:
                contains_crm_colname = True
                break
        # reject if the combination does not include a CRM colname
        if not contains_crm_colname:
            return

    if measure_weight(combination) < settings['min_weight_to_be_considered']:
        return
    elif measure_mean_weight(combination) < settings['overall_min_mean_weight']:
        return
    elif measure_mean_weight(combination) < settings['min_mean_weight'] and len(combination) > settings['min_len_for_min_mean_weight']:
        return
    else:
        return combination

def get_chunksize(number_of_combinations):
    """
    Computing the optimal chunksize for multiprocessing with respect
    to the provided settings. If chunksize is too small or too high,
    multiprocessing might be inefficient.
    """
    if number_of_combinations < settings['num_cores']:
        return 1

    chunksize = math.ceil(number_of_combinations/settings['num_cores'])

    if chunksize > settings['max_chunksize_per_processor']:
        return settings['max_chunksize_per_processor']
    else:
        return chunksize

def create_and_filter_colnames(colnames, L, number_of_combinations):
    """
    Creates all combinations for a given lengths and filteres the tuples based on the provided thresholds.
    """
    filtered_combinations = []

    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(eval_colname_for_filtering, itertools.combinations(colnames, L), chunksize=(get_chunksize(number_of_combinations))*3):
            if i is not None:
                filtered_combinations.append(i)
        pool.close()
        pool.join()

    return filtered_combinations

# Input a pandas seriesf
def get_entropy(data):
    """
    Calculates the entropy via Scipy.stats for provided data
    """
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy
    # scipy.stats.entropy calculates the entropy of a distribution for given probability values.
    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0)
    return entropy


#################################### Treat 2nd class identifier ####################################

# Explanation: "the shortcuts based on + (including the implied use in sum) are, of necessity,
# O(L**2) when there are L sublists -- as the intermediate result list keeps getting longer,
# at each step a new intermediate result list object gets allocated, and all the items in the
# previous intermediate result must be copied over (as well as a few new ones added at the end).
# So (for simplicity and without actual loss of generality) say you have L sublists of I items each:
# the first I items are copied back and forth L-1 times, the second I items L-2 times, and so on;
# total number of copies is I times the sum of x for x from 1 to L excluded, i.e., I * (L**2)/2."
#
# Source: https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
#
def get_most_common_colname_in_list_of_lists(combinations):
    c = count_colname_in_list_of_lists(combinations)
    return c[0][0]

def get_index_of_element_in_list_of_lists(element, list_of_lists):
    for index,item in enumerate(list_of_lists):
        if element in set(item):
            return index

    # if element not included in any list within the list, we weight high
    return len(list_of_lists)

def get_most_common_colname_in_combination(combination, colnames_popularity_sorted):
    """
    Selecting an element (column) from a combination based on the
    highest occurance through its sorted column popularity.
    """
    most_common_colname_index = len(colnames_popularity_sorted)
    most_common_colname_name = ""
    for element in combination:
        index = get_index_of_element_in_list_of_lists(element, colnames_popularity_sorted)
        #print("colname {0} with size {1}".format(colname_id_mapping[element], df[colname_id_mapping[element]].size))
        if index < most_common_colname_index and df[colname_id_mapping[element]].size > 2:
            most_common_colname_index = index
            most_common_colname_name = element

    return most_common_colname_name

def count_colname_in_list_of_lists(combinations):
    """
    Given is a list of tuples, we count how often each
    element is represented and return the Counter
    """
    merged = list(itertools.chain(*combinations))
    c = collections.Counter(merged)
    # c = [('Jellicle', 6), ('Cats', 5), ('And', 2)]
    return c

def get_colname_with_highest_cardinality(combination):
    """
    Selecting an element (column) from a combination based on the
    highest cardinality.
    """
    highest_cardinality_value = 0
    highest_cardinality_colname = ""

    for colname in combination:
        cardinality = statistics['cardinality'].index(colname)
        if  cardinality > highest_cardinality_value:
            highest_cardinality_value = cardinality
            highest_cardinality_colname = colname

    return colname

def define_treatment_type(sample, cardinality):
    """
    Defining a treatment type for a combination based on the provided cardinality and
    a data sample and the defined thresholds in the settings.
    """
    if not settings['overwrite_treatment'] == "no":
        return settings['overwrite_treatment']
    sample_is_numeric = str(sample).isdigit()

    if (sample_is_numeric and cardinality < settings['upper_threshold_perturbation']):
        return "perturbation"
    elif (not sample_is_numeric and cardinality < settings['upper_threshold_perturbation']):
        return "suppression"
    else:
        return "compartmentation"
    #if (not sample_is_numeric and cardinality < settings['upper_threshold_generalization']) or (cardinality > 0 and cardinality < settings['upper_threshold_suppression']):
    #    return "suppression"
    #elif (sample_is_numeric and cardinality < settings['upper_threshold_perturbation'] and cardinality > settings['upper_threshold_generalization']):
    #    return "perturbation"
    #elif cardinality > settings['upper_threshold_suppression'] and cardinality < settings['upper_threshold_generalization']:
    #    return "generalization"
    #elif cardinality > settings['upper_threshold_generalization']:
    #    return "compartmentation"
    #elif False:
    #    return "perturbation"

def find_closest_lower_upper(value):
    """
    finding the closest lower and closest upper end for a given one value
    """
    evaluating_value = int(value)
    bigger = []
    smaller = []
    for i in numeric_series_list:
        if i > evaluating_value:
            bigger.append(i)
        elif i < evaluating_value:
            smaller.append(i)

    if not smaller:
        closest_lower = evaluating_value
    else:
        closest_lower = max(smaller)

    if not bigger:
        closest_upper = evaluating_value
    else:
        closest_upper = min(bigger)

    return {"closest_upper": closest_upper, "closest_lower":closest_lower}

def find_closest_value(value):
    """
    Find closest value for a given one
    """
    if not str(value).isdigit():
        return {value: numpy.nan}

    closest = find_closest_lower_upper(value)

    distance_lower = int(value)-int(closest["closest_lower"])
    distance_upper = int(closest["closest_upper"])-int(value)
    if distance_lower > distance_upper:
        return {int(value): closest["closest_lower"]}
    else:
        return {int(value): closest["closest_upper"]}

def create_perturbed_value_smarter(values,colname_for_treatment,representatives,combination):
    """
    Creating a smarter perturbed value
    """
    condition_set = {}

    evaluating_value = values[combination.index(colname_for_treatment)]

    if not str(evaluating_value).isdigit():
        return {evaluating_value: numpy.nan}
    else:
        evaluating_value = int(evaluating_value)

    for index,value in enumerate(values):
        if combination[index] == colname_for_treatment:
            continue
        condition_set[combination[index]] = value

    unique = representatives.loc[representatives['count']==1]
    non_unique = representatives.loc[representatives['count']>1]
    non_unique_with_same_values = representatives

    for index,element in enumerate(combination):
        if element == colname_for_treatment:
            continue
        non_unique_with_same_values = non_unique_with_same_values.loc[non_unique_with_same_values[element] == values[index]]

    series_list = list(non_unique_with_same_values[colname_for_treatment].unique())
    # test if we evaluate only digits
    #numeric_series_list = filter(op.isNumberType, series_list)
    numeric_series_list = [s for s in series_list if str(s).isdigit()]

    #if not numeric_series_list or not str(numeric_series_list[0]).isdigit():
    if not numeric_series_list:
        return {evaluating_value: numpy.nan}

    bigger = []
    smaller = []
    for i in numeric_series_list:
        i = int(i)
        if i > evaluating_value:
            bigger.append(i)
        elif i < evaluating_value:
            smaller.append(i)

    if not smaller:
        closest_lower = evaluating_value
    else:
        closest_lower = max(smaller)

    if not bigger:
        closest_upper = evaluating_value
    else:
        closest_upper = min(bigger)

    distance_lower = int(evaluating_value)-int(closest_lower)
    distance_upper = int(closest_upper)-int(evaluating_value)

    if distance_lower == 0 or distance_upper == 0:
        return {evaluating_value: numpy.nan}

    if distance_lower > distance_upper:
        return {evaluating_value: closest_upper}
    else:
        return {evaluating_value: closest_lower}

def create_generalized_value(value):
    """
    Creating a generalized value by finding the closest neighbors
    """
    if not str(value).isdigit():
        return {value: numpy.nan}

    closest = find_closest_lower_upper(value)

    return {int(value): "{0}-{1}".format(int(closest["closest_lower"]), int(closest["closest_upper"]))}


def suppress_combination(representatives, colname_for_treatment):
    """
    This method executes the local suppression of specified rows for a given column.
    Those values will be replaced with null values in the data series
    """
    unique = representatives.loc[representatives['count']==1]
    rows_to_be_anonymized = unique[colname_for_treatment].values
    #before_null_counter = df[colname_for_treatment].isnull().sum()
    length_of_treated_series = df[colname_for_treatment].count()
    #print("Eval: {0} > {1}".format(len(rows_to_be_anonymized),(df[colname_for_treatment].count()/100*99)))
    if len(rows_to_be_anonymized) > (length_of_treated_series/100*99):#we check if we will replace nearly 99%
        df[colname_for_treatment] = numpy.nan
        return length_of_treated_series

    replacement_dict = {}
    for value in rows_to_be_anonymized:
        replacement_dict[value] = numpy.nan

    with multiprocessing.Pool(settings['num_cores']) as pool:
        mp_array = pool.imap_unordered(partial(replace_dict_in_series,replacement_dict=replacement_dict), numpy.array_split(df[colname_for_treatment],settings['num_cores']))
        #mp_array = pool.imap_unordered(partial(suppress_combination_partial,rows_to_be_anonymized=rows_to_be_anonymized), numpy.array_split(df[colname_for_treatment],settings['num_cores']))
        pool.close()
        pool.join()

    df[colname_for_treatment] = pd.concat(mp_array)
    #after_null_counter = df[colname_for_treatment].isnull().sum()
    #return (after_null_counter - before_null_counter)
    return len(replacement_dict)

def generalize_all_supersets_of_combination_partial(combination, colnames_popularity_sorted):
    """
    Generalizing all supersets of a specific combination.
    """
    representatives = df.groupby(resolve_id_mapping_combinations(combination), sort=False).size().reset_index().rename(columns={0:'count'})
    quasi_identifier_coverage = representatives.loc[representatives['count']==1]['count'].count()
    colname_id_for_treatment = get_most_common_colname_in_combination(combination, colnames_popularity_sorted)
    colname_for_treatment = colname_id_mapping[colname_id_for_treatment]

    # sip the combination if it is already resolved and no unique rows are left
    if quasi_identifier_coverage < 1:
        return 0

    return generalize_combination(representatives, colname_for_treatment,combination,with_multiprocessing=False)

def generalize_all_supersets_of_combination(combination, colnames_popularity_sorted):
    """
    Generalizing all supersets of a provided combination through partially multiprocessing.
    """
    replaced_values = 0
    all_supersets_of_combination = []
    colnames = list(df.columns.values)

    for L in range(len(combination),len(colnames)):
        all_supersets_of_combination += [item for item in itertools.combinations(range(len(colnames)), L) if set(item).issuperset(set(combination))]

    all_filtered_supersets_of_combination = []
    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(eval_colname_for_filtering, all_supersets_of_combination, chunksize=(get_chunksize(len(all_supersets_of_combination)))*3):
            if i is not None:
                all_filtered_supersets_of_combination.append(i)
        pool.close()
        pool.join()

    print("Generalizing {0}".format(combination))

    if not all_filtered_supersets_of_combination:
        return 0

    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(partial(generalize_all_supersets_of_combination_partial,colnames_popularity_sorted=colnames_popularity_sorted), all_filtered_supersets_of_combination, chunksize = get_chunksize(len(all_supersets_of_combination))):
            replaced_values += i
        pool.close()
        pool.join()

    return replaced_values

def replace_dict_in_series(partial_series, replacement_dict):
    """
    Replacing provided elements as dictionary in a data series.
    """
    #return partial_series.replace(to_replace=replacement_dict)
    # more efficient than replace function of pandas
    # https://stackoverflow.com/questions/41985566/pandas-replace-dictionary-slowness
    return partial_series.map(lambda x: replacement_dict.get(x,x))


def generalize_combination(representatives, colname_for_treatment, combination, with_multiprocessing=True):
    """
    Generalizing specific rows for a specified column. Depending on the setting either multiprocessed
    or single processed if already part of multiprocessing.
    """
    global numeric_series_list

    unique = representatives.loc[representatives['count']==1]
    rows_to_be_anonymized = unique[colname_for_treatment].values
    numeric_series = pd.to_numeric(df[colname_for_treatment], errors='coerce')
    numeric_series = numeric_series.dropna(how='any').unique()
    numeric_series_list = list(numeric_series)

    replacement_dict = {}

    if with_multiprocessing:
        with multiprocessing.Pool(settings['num_cores']) as pool:
            for ruleset in pool.imap_unordered(create_generalized_value, rows_to_be_anonymized, chunksize = get_chunksize(len(rows_to_be_anonymized))):
                for key,value in ruleset.items():
                    replacement_dict[key] = value

            pool.close()
            pool.join()

        with multiprocessing.Pool(settings['num_cores']) as pool:
            mp_array = pool.imap_unordered(partial(replace_dict_in_series,replacement_dict=replacement_dict), numpy.array_split(df[colname_for_treatment],settings['num_cores']))
            pool.close()
            pool.join()

        df[colname_for_treatment] = pd.concat(mp_array)
    else:
        for row in rows_to_be_anonymized:
            for key,value in create_generalized_value(row).items():
                replacement_dict[key] = value

        df[colname_for_treatment] = replace_dict_in_series(df[colname_for_treatment], replacement_dict)

    return len(replacement_dict)


def perturbe_combination(representatives, colname_for_treatment, combination, with_multiprocessing=True):
    """
    Pertubate specific rows for a specified column. Depending on the setting either multiprocessed
    or single processed if already part of multiprocessing.
    """
    global numeric_series_list

    unique = representatives.loc[representatives['count']==1]
    #rows_to_be_anonymized = unique[colname_for_treatment].values
    #numeric_series = pd.to_numeric(df[colname_for_treatment], errors='coerce')
    #numeric_series = numeric_series.dropna(how='any').unique()
    #numeric_series_list = list(numeric_series)

    multiindex_rows_to_be_anonymized = unique[combination].values

    replacement_dict = {}
    #replacement_dict[colname_for_treatment] = {}
    negative_score = 0
    if with_multiprocessing:
        with multiprocessing.Pool(settings['num_cores']) as pool:
            for ruleset in pool.imap_unordered(partial(create_perturbed_value_smarter,colname_for_treatment=colname_for_treatment,representatives=representatives,combination=combination), multiindex_rows_to_be_anonymized, chunksize = get_chunksize(len(multiindex_rows_to_be_anonymized))):
            #for ruleset in pool.imap_unordered(find_closest_value, rows_to_be_anonymized, chunksize = get_chunksize(len(rows_to_be_anonymized))):
                for key,value in ruleset.items():
                    replacement_dict[key] = value

            pool.close()
            pool.join()

        with multiprocessing.Pool(settings['num_cores']) as pool:
            mp_array = pool.imap_unordered(partial(replace_dict_in_series,replacement_dict=replacement_dict), numpy.array_split(df[colname_for_treatment],settings['num_cores']))
            pool.close()
            pool.join()

        df[colname_for_treatment] = pd.concat(mp_array)
    else:
        for row in multiindex_rows_to_be_anonymized:
        #for row in rows_to_be_anonymized:
            ruleset = create_perturbed_value_smarter(row,colname_for_treatment,representatives,combination)
            #for key,value in find_closest_value(row).items():
            for key,value in ruleset.items():
                replacement_dict[key] = value

        df[colname_for_treatment] = replace_dict_in_series(df[colname_for_treatment], replacement_dict)

    #df[colname_for_treatment] = df[colname_for_treatment].replace(to_replace=replacement_dict)
    # TODO: negative_score subtract from data score
    return len(replacement_dict)


def treat_quasi_identifier_combination(combination, colnames_popularity_sorted):
    """
    Handling the treatment of one particular selected combination.
    This includes assessing with element (column) of this combination
    shall be treated. How to treat the selected column and executing
    the treatment.
    """
    representatives = df.fillna(-1).groupby(resolve_id_mapping_combinations(combination), sort=False).size().reset_index().rename(columns={0:'count'})
    quasi_identifier_coverage = representatives.loc[representatives['count']==1]['count'].count()

    # sip the combination if it is already resolved and no unique rows are left
    if quasi_identifier_coverage < 1:
        logging.info("Skipping combi: {0} since the quasi_identifier_coverage is {1}".format(resolve_id_mapping_combinations(combination),quasi_identifier_coverage))
        return {"colname_id_for_treatment":None,"amount_of_altered_rows":0,"series_length":0}

    quasi_identifier_coverage_sample = representatives.loc[representatives['count']==1].iloc[0]
    cardinality = (quasi_identifier_coverage / len(df.index) * 100)

    #colname_for_treatment = get_colname_with_highest_cardinality(combination)
    colname_id_for_treatment = get_most_common_colname_in_combination(combination, colnames_popularity_sorted)
    colname_for_treatment = colname_id_mapping[colname_id_for_treatment]

    treatment_type = define_treatment_type(quasi_identifier_coverage_sample[colname_for_treatment], cardinality)
    logging.info("Treating combi: {0} with cardinality {1} through {2}".format(resolve_id_mapping_combinations(combination), cardinality, treatment_type))
    series_length = df[colname_for_treatment].count()

    # choose suppression if cardinality respects given thresholds or if column is string based (categorical) and cardinality is too low for compartmentation
    if treatment_type == "suppression":
        amount_of_altered_rows = suppress_combination(representatives, colname_for_treatment)
        logging.info("---> Suppressing {0}/{1} ({2}%) values in column {3}".format(amount_of_altered_rows, series_length,math.ceil(amount_of_altered_rows/series_length*100), colname_for_treatment))

    elif treatment_type == "generalization":
        # executing the generalization for the already selected column
        amount_of_altered_rows = generalize_combination(representatives, colname_for_treatment,combination)
        # since we just alter and not remove values, we have to reevaluate all supersets of this mUCC for potential new 2n class identifier
        amount_of_altered_rows += generalize_all_supersets_of_combination(combination, colnames_popularity_sorted)
        logging.info("---> Generalizing {0}/{1} ({2}%) values in column {3}".format(amount_of_altered_rows, series_length, math.ceil(amount_of_altered_rows/series_length*100), colname_for_treatment))

    elif treatment_type == "compartmentation":
        amount_of_altered_rows = 0
        # just collect those combinations, we will process them later on in a compressed way to reduce data redundancy
        outstanding_combinations_for_compartmentation.append(combination)
        logging.info("---> Compartment of column {0}".format(resolve_id_mapping_combinations(combination)))

    elif treatment_type == "perturbation":
        # executing the perturbation for the already selected column
        amount_of_altered_rows = perturbe_combination(representatives, colname_for_treatment, resolve_id_mapping_combinations(combination))
        # since we just alter and not remove values, we have to reevaluate all supersets of this mUCC for potential new 2n class identifier
        logging.info("---> Perturbation {0}/{1} ({2}%) values in column {3}".format(amount_of_altered_rows, series_length, math.ceil(amount_of_altered_rows/series_length*100), colname_for_treatment))

    #elif treatment_type == "delete":
    else:
        print("WARNING: no treatment was selected for combination: {0} with cardinality {1}. treatment_type={2}".format(resolve_id_mapping_combinations(combination), cardinality, treatment_type))

    return {"colname_id_for_treatment":colname_id_for_treatment,
            "amount_of_altered_rows":amount_of_altered_rows,
            "series_length":series_length}

def get_combinations_effected_by_teatment(colname):
    """
    Evaluate which tuples of quasi identifiers become obsolete
    after treating a given column
    """
    combinations = []
    for combination in quasi_identifier_combinations:
        if colname in combination:
            combinations.append(combination)

    return combinations


def build_graph_for_tuple_combination(combination, naming, directed_graph=False, resolve_id_mapping=False, visualize=False, add_column_as_node=False):
    """
    building a graphical representation for a given tuple set
    """
    if directed_graph:
        G = networkx.DiGraph()
    else:
        G = networkx.Graph()
    for node_tuple in combination:
        if resolve_id_mapping or add_column_as_node:
            node_tuple = resolve_id_mapping_combinations(node_tuple)
        G.add_edges_from(itertools.product(node_tuple, node_tuple))

    if add_column_as_node:
        for key,colname in colname_id_mapping.items():
            G.add_node(colname)

    if visualize:
        networkx.draw(G, with_labels = True)
        plt.savefig('images/'+naming+'.png')
        plt.clf()

    return G

def draw_circle_around_clique(clique,coords,colors,hatches):
    """
    For visualization purposes, mark maximal cliques within a graph
    """
    dist     = 0
    tmp_dist = 0
    center   = [0 for i in range(2)]
    color    = next(colors)
    for a in clique:
        for b in clique:
            tmp_dist = (coords[a][0]-coords[b][0])**2+(coords[a][1]-coords[b][1])**2
            if tmp_dist > dist:
                dist = tmp_dist
                for i in range(2):
                    center[i] = (coords[a][i]+coords[b][i])/2
    rad = dist**0.5/2
    #cir = plt.Circle((center[0],center[1]),radius=rad*1.3,fill=False,color=color,hatch=next(hatches))
    cir = plt.Circle((center[0],center[1]),radius=rad*1.3,fill=False,color=color)
    plt.gca().add_patch(cir)
    plt.axis('scaled')
    # return color of the circle, to use it as the color for vertices of the cliques
    return color

def execute_compartmentation():
    """
    Compartmentation logic:
    By finding intersections inbetween sets and then sprinkle all colnames around the
    intersection points except those which are not supposed to go along
    """
    if settings['create_visualize'] == 'yes':
        visualize=True
    else:
        visualize=False

    G1 = build_graph_for_tuple_combination(outstanding_combinations_for_compartmentation, "tuples_for_compartmentation", resolve_id_mapping=True, add_column_as_node=True, visualize=visualize)
    missing_edges = [pair for pair in itertools.combinations(G1.nodes(), 2) if not G1.has_edge(*pair)]

    # be aware, when resolve_id_mapping was set to true for G1, then it has to be False for this one otherwise
    # a KeyError for the resolving will raise
    G2 = build_graph_for_tuple_combination(missing_edges, "inverse_tuples_for_compartmentation", visualize=visualize)
    # determining all maximal cliques in the reverse tuples as sanitized and compartments
    sanitized_compartments = list(networkx.find_cliques(G2))

    if visualize:
        coords = networkx.spring_layout(G2)
        colors = itertools.cycle('bgrcmyk')
        hatches = itertools.cycle('/\|-+*')
        networkx.draw(G2,pos=coords)
        for clique in sanitized_compartments:
            networkx.draw_networkx_nodes(G2,pos=coords,nodelist=clique,node_color=draw_circle_around_clique(clique,coords,colors,hatches))
            networkx.draw_networkx_labels(G2,pos=coords, font_size=16)#, font_color='r')

        plt.savefig('images/cliques.png')
        plt.clf()

    print("There are {0} compartments".format(len(sanitized_compartments)))
    logging.info("There are {0} compartments".format(len(sanitized_compartments)))
    return sanitized_compartments


def treat_quasi_identifier_combinations():
    """
    This method orchestrates the tratment of identified
    quasi identifiers (esp. 2nd class identifiers)
    """
    sanitized_quasi_identifier_combinations = []
    global outstanding_combinations_for_compartmentation
    outstanding_combinations_for_compartmentation = []
    treatment_counter= 0
    total_value_counter=0
    colnames_popularity = count_colname_in_list_of_lists(quasi_identifier_combinations)

    colnames_popularity_sorted = colnames_popularity.most_common()

    for combination in quasi_identifier_combinations:
        # we might skip already treated columns or rather already sanitized_quasi_identifier_combinations
        # however, since we only partially sanitize column, lets recheck then better
        #if combination in sanitized_quasi_identifier_combinations:
        #    continue

        result = treat_quasi_identifier_combination(combination, colnames_popularity_sorted) # returns treated colname
        treatment_counter += result["amount_of_altered_rows"]
        total_value_counter += result["series_length"]
        #select all combinations which include one of the colnames for reevaluation
        #if colname not None:
        #   sanitized_quasi_identifier_combinations += get_combinations_effected_by_teatment(result["colname_id_for_treatment"])
    if treatment_counter>0 and total_value_counter>0:
        percentage = math.ceil(treatment_counter/total_value_counter*100)
    else:
        percentage = 0

    logging.info("Treated {0} of {1} ( {2} %) values".format(treatment_counter,total_value_counter,percentage))

    with open(get_filename("treatment"), "a") as myfile:
        myfile.write(str(treatment_counter)+","+str(total_value_counter)+","+str(settings["amount_of_columns"])+"\n")

    return execute_compartmentation()

#################################### Identify 1st & 2nd class identifier ####################################

def identify_1st_identifier(colname):
    """
    Evaluating a provided column name whether it may be
    considered as 1st class identifier based on the
    defined threshold in the settings. This threshold may
    include an error rate for null values etc.
    """
    global statistics
    representatives = df.fillna(-1).groupby(colname, sort=False).size().reset_index().rename(columns={0:'count'})
    unique_entries = representatives.loc[representatives['count']==1]['count'].count()
    coverage_of_uniques = unique_entries / ( len(df.index) - df[colname].isnull().sum() )

    # In the context of databases, cardinality refers to the uniqueness of data values contained in a column.
    # High cardinality means that the column contains a large percentage of totally unique values.
    # Low cardinality means that the column contains a lot of repeats in its data range.
    # Source: https://www.techopedia.com/definition/18/cardinality-databases
    cardinality = len(df[colname].unique()) / ( len(df.index) - df[colname].isnull().sum() )
    entropy = get_entropy(df[colname].dropna(axis=0, how='all'))

    stats = [entropy,
             cardinality,
             unique_entries,
             coverage_of_uniques]

    if (coverage_of_uniques* 100) > settings['threshold_id_column']:# or (cardinality* 100) > settings['threshold_cardinality']:
        return({"colname":colname, "identifier":True, "stats":stats})
    else:
        return({"colname":colname, "identifier":False, "stats":stats})

def resolve_id_mapping_combinations(ids):
    """
    A helper class for resolving a ID set back to the
    original column names for better comprehension
    e.g. logging or printing
    """
    #mapping ids back to strings
    mapped_numbers = []
    for number in ids:
        mapped_numbers.append(colname_id_mapping[number])

    return mapped_numbers

def identify_2nd_identifier(subset):
    """
    Evaluating a provided subset whether it may be considered
    as 2nd class identifier based on a cardinality threshold
    provided in the settings
    """
    representatives = df.groupby(resolve_id_mapping_combinations(subset), sort=False).size().reset_index().rename(columns={0:'count'})
    quasi_identifier_coverage = representatives.loc[representatives['count']==1]['count'].count()
    del representatives

    cardinality = (quasi_identifier_coverage / len(df.index)* 100)
    if  quasi_identifier_coverage > 0:
        #if cardinality > settings['upper_threshold_suppression']:
        return subset

def search_2nd_class_identifier(colnames, statistics, first_class_identifiers):
    """
    This method implements the multiprocessing search for 2nd class identifiers
    considering all possible combinations of all lengths for the given column names
    """
    iterations_run = 1
    list_of_lens = {}

    global colname_id_mapping
    colname_id_mapping = {}

    # using ids for combinations instead of strings due to size
    for index,colname in enumerate(colnames):
        colname_id_mapping[index] = colname

    colnames_without_first_identifiers = [x for x in colnames if x not in first_class_identifiers]

    global colnames_weight
    colnames_weight = {}

    for index, row in statistics.iterrows():
        colnames_weight[index] = row['entropy']

    combination_eval_count = 0
    total_combis_count = 0

    # finding 2nd class identifier through out the all combinations of all lengths
    for L in range(2, len(colnames)+1):

        if settings['find_2nd_class_identifier_exact'] == 'yes':
            logging.info("--> {0} / {1} (Exact Mode)".format(L,len(colnames)))
            combinations_for_evaluation = list(itertools.combinations(range(len(colnames)), L))
        else:
            total_combination_number = ncr(len(colnames),L)
            combinations_for_evaluation = create_and_filter_colnames(range(len(colnames)), L, total_combination_number)
            total_combis_count += total_combination_number
            combination_eval_count += len(combinations_for_evaluation)
            savings = math.ceil(len(combinations_for_evaluation)/total_combination_number*10000)/100
            logging.info("--> {0} / {1} (Filtered Mode with {2} % for {3} combis )".format(L,len(colnames),savings,total_combination_number ))

        if len(combinations_for_evaluation) < 1:
            logging.info("Break, there are none filtered_colnames left to evaluate.")
            break

        count_eval_combinations_no_supersets = 0
        list_of_lens[L] = []

        with multiprocessing.Pool(settings['num_cores']) as pool:
            for i in pool.imap_unordered(identify_2nd_identifier, combinations_for_evaluation):
                if i is not None:
                    quasi_identifier_combinations.append(i)

            pool.close()
            pool.join()

        logging.info("-----> Currently {0} mpmUCCs found".format(len(quasi_identifier_combinations)))
        iterations_run += 1

    return({"iterations_run":iterations_run,
            "combination_eval_count":combination_eval_count,
            "total_combis_count":total_combis_count,
            "quasi_identifier_combinations": quasi_identifier_combinations})

def identify_treat_1st_2nd_class_identifier(df_local, global_iteration, retrieve=None):
    """
    Orchestrate the overall identification of 1st class identifiers
    as well as 2nd class identifiers. Those quasi identifiers will
    be stored in a global variable. Additionally, the column names
    (strings) will be mapped to integer to reduce the memory size
    when building all combinations. This can get quickly large
    e.g. for ~80 columns --> nearly 5 billion combinations considering
    all possible lengths.
    """
    global quasi_identifier_combinations
    quasi_identifier_combinations = []
    global colname_id_mapping
    colname_id_mapping = {}
    global df
    df = df_local
    df_original = df_local
    global statistics
    statistics = pd.DataFrame(columns=('colname', 'entropy', 'entropy_rounded', 'cardinality',
                                        'occurance_in_ucc', 'mean_of_ucc_length', 'allowed_combinations',
                                        'mean_of_allowed_combinations_length', 'unique_entries',
                                        'coverage_of_uniques'))
    statistics = statistics.set_index(['colname'])

    # cleanup df
    df = df.dropna(axis=1, how='all')
    colnames = list(df.columns.values)
    first_class_identifiers = []

    # finding & removing first class identifier
    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(identify_1st_identifier, colnames):
            statistics.set_value(i["colname"], 'entropy', i["stats"][0])
            statistics.set_value(i["colname"], 'entropy_rounded', round(i["stats"][0],2))
            statistics.set_value(i["colname"], 'cardinality', i["stats"][1])
            statistics.set_value(i["colname"], 'unique_entries', i["stats"][2])
            statistics.set_value(i["colname"], 'coverage_of_uniques', i["stats"][3])

            if i["identifier"]:
                logging.info("Dropping 1st class identifier: {0}".format(i["colname"]))
                first_class_identifiers.append(i["colname"])
                del df[i["colname"]] # treatment of 1nd class identifier

        pool.close()
        pool.join()

    # cleanup df
    df = df.dropna(axis=1, how='all')
    colnames = list(df.columns.values)

    if len(colnames) < 2:
        logging.info("Stop: There is only 1 col left, no need for 2nd class identifier detection")
        return({"combis_count": 0, "number_of_ucc":0, "iterations_run": 0, "identifiers":[],"combination_eval_count":0})

    global length_of_colnames
    length_of_colnames = len(colnames)+1

    filename = "cache/"+str(settings["amount_of_columns"])+"_mpmUCCs.txt"
    my_file = Path(filename)
    if my_file.is_file() and settings["cache_2nd_class_identifiers"]:
        stats = {}
        with open (filename, 'rb') as fp:
            quasi_identifier_combinations = pickle.load(fp)
            stats["quasi_identifier_combinations"]= quasi_identifier_combinations
        global colname_id_mapping
        colname_id_mapping = {}
        for index,colname in enumerate(colnames):
            colname_id_mapping[index] = colname
    else:
        identification_start_time = time.time()
        stats = search_2nd_class_identifier(colnames, statistics, first_class_identifiers)
        with open(get_filename("search_quasi_identifier_timing"), "a") as myfile:
            myfile.write(str((time.time() - identification_start_time))+","+str(settings["amount_of_columns"])+"\n")
        with open(filename,"wb") as fp:
            pickle.dump(quasi_identifier_combinations, fp)

    number_of_ucc = len(stats["quasi_identifier_combinations"])
    logging.info("Found {0} mpmUCCs".format(number_of_ucc))

    logging.info("----- Starting Treatment of mpmUCCs -----")

    if settings["create_visualize"] == "yes":
        visualize=True
    else:
        visualize=False
    build_graph_for_tuple_combination(quasi_identifier_combinations, "quasi_identifier_combinations", directed_graph=True, resolve_id_mapping=True, visualize=visualize)

    with open("output/mpmUCCs_counter.txt","a") as myfile:
         myfile.write(str(len(quasi_identifier_combinations))+","+str(settings["amount_of_columns"])+"\n")

    if settings["analyze_results"] == 'yes':
        quasi_identifier_combinations_resolved = []
        for i in quasi_identifier_combinations:
            quasi_identifier_combinations_resolved.append(resolve_id_mapping_combinations(i))

        analyze_quasi_identifier_combinations(settings, colname_id_mapping, colnames, global_iteration, quasi_identifier_combinations_resolved, statistics, colnames_weight)

    if settings["do_treatment"] == 'yes':
        treatment_start_time = time.time()
        sanitized_compartments = treat_quasi_identifier_combinations()
        with open(get_filename("treatment_timing"), "a") as myfile:
            myfile.write(str((time.time() - treatment_start_time))+","+str(settings["amount_of_columns"])+"\n")

        df_sanitized = pd.DataFrame()
        if not sanitized_compartments:
            for compartment in sanitized_compartments:
                df_sanitized = df_sanitized.append(df[compartment], ignore_index=True)
        else:
            df_sanitized = df
        df_filenames = write_data(df_sanitized, global_iteration)

    if settings["measure_data_score"] == 'yes':
        score = estimate_data_quality(df_sanitized,colnames,settings['num_cores'],df_original)
        logging.info("Score: {0} with {1} columns".format(round(score,2), settings["amount_of_columns"]))

        with open(get_filename("scores"), "a") as myfile:
            myfile.write(str(round(score,2))+","+str(settings["amount_of_columns"])+"\n")

    if settings["measure_data_size"] == 'yes':
        size = 0
        df_size = df_sanitized.memory_usage(index=True)
        for colsize in df_size:
            size += colsize

        logging.info("Size: {0} with {1} columns".format(size, settings["amount_of_columns"]))

        with open(get_filename("size"), "a") as myfile:
            myfile.write(str(size)+","+str(settings["amount_of_columns"])+"\n")
        with open(get_filename("compartments"), "a") as myfile:
            myfile.write(str(len(sanitized_compartments))+","+str(settings["amount_of_columns"])+"\n")

    if settings["do_treatment"]=='yes' and settings["eval_treatment"]=='yes':
        logging.info("----- Double check if anonymized -----")
        quasi_identifier_combinations = []
        for index in range(1,len(sanitized_compartments)):
            df = pd.read_csv("output/"+str(global_iteration)+"_"+str(index)+"_sanitized_df.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')
            print("Loaded compartment with columns: {0}".format(list(df.columns.values)))
            local_stats = search_2nd_class_identifier(colnames, statistics, first_class_identifiers)
            logging.info("Found {0} mpmUCCs:".format(len(local_stats["quasi_identifier_combinations"])))
            for combi in local_stats["quasi_identifier_combinations"]:
                logging.info(resolve_id_mapping_combinations(combi))

    if retrieve == "stats":
        return({
            "combis_count": stats["total_combis_count"],
            "number_of_ucc": number_of_ucc,
            "iterations_run": stats["iterations_run"],
            "identifiers":quasi_identifier_combinations,
            "combination_eval_count":stats["combination_eval_count"]
            })
    elif retrieve == "df" and settings["do_treatment"] == 'yes':
        return df_filenames
    else:
        return

############################## Helper ##############################

def set_settings(settings_local):
    global settings
    settings = settings_local

def ncr(n,r):
    """
    Calculating the number of combinations for the binominal coefficient
    for the provided lengths elements (r) and length (n)
    """
    #n = length_of_colnames
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    result =  numer//denom
    return result

def write_data(df_sanitized, global_iteration):
    logging.info("Writing df to file..")
    filename = "output/"+str(settings["amount_of_columns"])+"_sanitized_df.csv.gz"
    df_sanitized.to_csv(filename, compression='gzip', sep=',', quotechar='"')
    return filename

def retrieve_data():
    """
    Reading data from gip csv file provided in the settings and
    storing it as global panda dataframe variable
    """
    global crm_colnames
    df = pd.read_csv(settings['data_path']+settings['source_data'],
                    compression='gzip', header=0, sep=',', quotechar='"',
                    nrows=settings['max_rows'])
    crm_colnames = [
                    "GivenName","Surname","Gender","NameSet","Title","StreetAddress","City","State","StateFull",
                    "ZipCode","Country","EmailAddress","Username","Password","TelephoneNumber","TelephoneCountryCode",
                    "Birthday","Age","NationalID","Color","Occupation","Company","BloodType","Kilograms","Centimeters"
                    ]
    return df

def retrieve_meta_data():
    """
    Reading meta data from gip csv file provided in the settings and
    storing it as global panda dataframe variable
    """
    global crm_colnames

    df = pd.read_csv(settings['data_path']+settings['source_data'],
                    compression='gzip', header=0, sep=',', quotechar='"',
                    nrows=10)
    crm_colnames = [
                    "GivenName","Surname","Gender","NameSet","Title","StreetAddress","City","State","StateFull",
                    "ZipCode","Country","EmailAddress","Username","Password","TelephoneNumber","TelephoneCountryCode",
                    "Birthday","Age","NationalID","Color","Occupation","Company","BloodType","Kilograms","Centimeters"
                    ]
    return df

############################## Main ##############################
def main():
    """
    Main functions orchestrating the major work
    """
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S_")
    logging.basicConfig(filename='log/'+str(datetime)+'panda.log', level=logging.INFO)

    df = retrieve_data()
    colnames = list(df.columns.values)

    if settings["amount_of_columns"] > len(colnames):
        settings["amount_of_columns"] = len(colnames)

    ##remove before flight
    #test = estimate_data_quality(df,colnames,settings['num_cores'])
    #print(test)
    #return

    print("Data fetched with {0} columns, processing now mit {1} cores".format(settings["amount_of_columns"],settings['num_cores']))
    start_time = time.time()

    identify_treat_1st_2nd_class_identifier(df[colnames[:settings["amount_of_columns"]]], settings["amount_of_columns"])
    logging.info("Execution took {0} seconds".format((time.time() - start_time)))

    with open(get_filename("timing"), "a") as myfile:
        myfile.write(str((time.time() - start_time))+","+str(settings["amount_of_columns"])+"\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
