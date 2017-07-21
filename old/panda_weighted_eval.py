import numpy
import pandas as pd
import os
import sys
import time
import multiprocessing
from contextlib import closing
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
# own
#from clustering import *
#from analyzer import *

sys.path.append('./')
#sys.path.append('../')

parser = argparse.ArgumentParser()

parser.add_argument("-ncol", "--columns", default = 12, help="Amount of columns to be considered", type=int)
parser.add_argument("-exact", "--exact", default = 'no', help="Determining 2nd class identifier exact instead of heuristic", type=str)
args = parser.parse_args()

settings = {
    'source_data': 'extended_user_data.csv.gz',
    'data_path': '../../',
    'max_rows': 500000, # set 0 for unlimited
    'threshold_supression_vs_compartment': 10, # in percent
    'threshold_id_column': 80, # in percent
    'threshold_cardinality': 30,
    'num_cores': (multiprocessing.cpu_count()-1),
    'amount_of_columns': args.columns,
    'find_2nd_class_identifier_exact': args.exact,
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
    return math.ceil(sum(colnames_weight[colname_id_mapping[x]] for x in column_combination))


def measure_mean_weight(column_combination):
    column_selection = [colnames_weight[colname_id_mapping[x]] for x in column_combination]
    if not column_selection:
        return 0
    estimated_mean = numpy.mean(column_selection)
    return math.ceil(estimated_mean)


def eval_colname_for_filtering(combination):
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


def create_and_filter_colnames(colnames, L):
    filtered_combinations = []

    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(eval_colname_for_filtering, itertools.combinations(colnames, L), chunksize =1000):
            if i is not None:
                filtered_combinations.append(i)
        pool.close()
        pool.join()

    return filtered_combinations

# Input a pandas seriesf
def get_entropy(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy
    # scipy.stats.entropy calculates the entropy of a distribution for given probability values.
    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0)
    return entropy

#################################### Identify & Treat 1st & 2nd class identifier ####################################


def identify_1nd_identifier(colname):
    global statistics
    representatives = df.groupby(colname, sort=False).size().reset_index().rename(columns={0:'count'})
    unique_entries = representatives.loc[representatives['count']==1]['count'].count()
    coverage_of_uniques = unique_entries / ( len(df.index) - df[colname].isnull().sum() )

    # In the context of databases, cardinality refers to the uniqueness of data values contained in a column.
    # High cardinality means that the column contains a large percentage of totally unique values.
    # Low cardinality means that the column contains a lot of “repeats” in its data range.
    # Source: https://www.techopedia.com/definition/18/cardinality-databases

    cardinality = len(df[colname].unique()) / ( len(df.index) - df[colname].isnull().sum() )
    entropy = get_entropy(df[colname].dropna(axis=0, how='all'))

    stats = [
                entropy,
                cardinality,
                unique_entries,
                coverage_of_uniques
            ]

    if (coverage_of_uniques* 100) > settings['threshold_id_column'] or (cardinality* 100) > settings['threshold_cardinality']:
        return({"colname":colname, "identifier":True, "stats":stats})
    else:
        return({"colname":colname, "identifier":False, "stats":stats})

def resolve_id_mapping_combinations(ids):
    #mapping ids back to strings
    mapped_numbers = []
    for number in ids:
        mapped_numbers.append(colname_id_mapping[number])

    return mapped_numbers

def identify_2nd_identifier(subset):
    representatives = df.groupby(resolve_id_mapping_combinations(subset), sort=False).size().reset_index().rename(columns={0:'count'})
    quasi_identifier_coverage = representatives.loc[representatives['count']==1]['count'].count()
    del representatives

    cardinality = (quasi_identifier_coverage / len(df.index)* 100)
    if  quasi_identifier_coverage > 0:
        if cardinality > settings['threshold_supression_vs_compartment']:
            return subset

def search_2nd_class_identifier(colnames, statistics, first_class_identifiers):
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
            combinations_for_evaluation = create_and_filter_colnames(range(len(colnames)), L)
            total_combination_number = ncr(len(colnames),L)
            total_combis_count += total_combination_number
            combination_eval_count += len(combinations_for_evaluation)
            savings = math.ceil(len(combinations_for_evaluation)/total_combination_number*10000)/100

            logging.info("--> {0} / {1} (Filtered Mode with {2} % for {3} combis )".format(L,len(colnames),savings,total_combination_number ))

        if len(combinations_for_evaluation) < 1:
            logging.info("there are none filtered_colnames")
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

    return(
                { "iterations_run":iterations_run,
                  "combination_eval_count":combination_eval_count,
                  "total_combis_count":total_combis_count
                })

def multi_processing_identify_partitioning(df,global_iteration):
    global quasi_identifier_combinations
    quasi_identifier_combinations = []
    global statistics
    global colname_id_mapping
    colname_id_mapping = {}
    statistics = pd.DataFrame(columns=('colname', 'entropy', 'entropy_rounded', 'cardinality',
                                        'occurance_in_ucc', 'mean_of_ucc_length', 'allowed_combinations',
                                        'mean_of_allowed_combinations_length', 'unique_entries',
                                        'coverage_of_uniques', 'cluster_size', 'assignment'))
    statistics = statistics.set_index(['colname'])

    # cleanup df
    df = df.dropna(axis=1, how='all')
    colnames = list(df.columns.values)

    first_class_identifiers = []

    # finding & removing first class identifier
    with multiprocessing.Pool(settings['num_cores']) as pool:
        for i in pool.imap_unordered(identify_1nd_identifier, colnames):
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

    stats = search_2nd_class_identifier(colnames, statistics, first_class_identifiers)

    number_of_ucc = len(quasi_identifier_combinations)
    logging.info("Found {0} mpmUCCs".format(number_of_ucc))

    return({
        "combis_count": stats["total_combis_count"],
        "number_of_ucc": number_of_ucc,
        "iterations_run": stats["iterations_run"],
        "identifiers":quasi_identifier_combinations,
        "combination_eval_count":stats["combination_eval_count"]
        })


############################## Helper ##############################

def ncr(n,r):
    #n = length_of_colnames
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    result =  numer//denom
    return result

def retrieve_data():
    global df
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

############################## Main ##############################
def main():
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S_")
    logging.basicConfig(filename='log/'+str(datetime)+'panda_weight_eval.log', level=logging.INFO)

    #loading csv data
    df = retrieve_data()

    colnames = list(df.columns.values)
    timing_results = pd.DataFrame(columns=('colcount', 'timer', 'combis_count', 'number_of_ucc', 'iterations_run','combination_eval_count' ))


    for i in range(2,len(colnames)):
        logging.info("-------------------- ROUND {0} --------------------".format(i))
        start_time = time.time()
        settings["amount_of_columns"] = i
        result = multi_processing_identify_partitioning(df[colnames[:i]],i)
        global_timer = (time.time() - start_time)
        logging.info("Execution took {0} seconds".format(global_timer))

        uccs_with_identifier_lengths = [len(i) for i in result["identifiers"]]
        uccs_with_identifier_weights = [measure_weight(i) for i in result["identifiers"]]
        timing_results.set_value(i, 'timer', round(global_timer,2))
        timing_results.set_value(i, 'combis_count', result["combis_count"])
        timing_results.set_value(i, 'number_of_ucc', result["number_of_ucc"])
        timing_results.set_value(i, 'iterations_run', result["iterations_run"])
        timing_results.set_value(i, 'combination_eval_count', result["combination_eval_count"])

        if uccs_with_identifier_lengths:
            max_ucc_size = max(uccs_with_identifier_lengths)
            for m in range(2, max_ucc_size):
                timing_results.set_value(i, 'ucc_size_'+str(m), uccs_with_identifier_lengths.count(m))

        if uccs_with_identifier_weights:
            unique_weights = []
            for i in uccs_with_identifier_weights:
                if i not in unique_weights:
                    unique_weights.append(i)
            for index, weight in enumerate(unique_weights):
                timing_results.set_value(i, 'ucc_weight_'+str(index), weight)
                timing_results.set_value(i, 'ucc_weight_'+str(index)+'_count', uccs_with_identifier_weights.count(weight))

    timing_results.to_csv("output/"+str(datetime)+"eval_timing_results.csv", sep=';', quotechar='"')

multiprocessing.freeze_support()
main()
