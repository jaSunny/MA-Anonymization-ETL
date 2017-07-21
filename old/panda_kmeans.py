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
# own
from clustering import *
from analyzer import *

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
    'find_2nd_class_identifier_exact': args.exact
}



#################################### helper ########################################################

def get_sample_of_colnames(colnames, sample_size):
    return numpy.random.choice(colnames, size=sample_size)
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

    cardinality = (quasi_identifier_coverage / len(df.index)* 100)
    if  quasi_identifier_coverage > 0:
        if cardinality > settings['threshold_supression_vs_compartment']:
            return(subset)

    return

def is_covered_by_minimal_identifier(combination):
    for quasi_identifier in quasi_identifier_combinations:
        if set(combination).issuperset(quasi_identifier):
            return False

    return identify_2nd_identifier(combination)

def get_combinations_for_evaluation(clustered_columns, colnames, k_means_input, L):
    if settings['find_2nd_class_identifier_exact'] == 'yes':
        logging.info("Determining exact 2nd class identifier")
        return itertools.combinations(range(0,len(colnames)), L)
    else:
        logging.info("Determining approx. 2nd class identifier due to representations")
        sample_with_one_item_per_cluster = choose_sample_from_each_cluster(clustered_columns, 1)

        combinations = []
        for item in sample_with_one_item_per_cluster:
            combinations += choose_samples_from_other_cluster(clustered_columns, L, item)

        return combinations

def multi_processing_identify_partitioning(df,global_iteration):
    global quasi_identifier_combinations
    quasi_identifier_combinations = []
    global found_less_than_one
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

    k_means_input = []
    # finding & removing first class identifier
    with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
        for i in pool.imap_unordered(identify_1nd_identifier, colnames):
            statistics.set_value(i["colname"], 'entropy', i["stats"][0])
            statistics.set_value(i["colname"], 'entropy_rounded', round(i["stats"][0],2))
            statistics.set_value(i["colname"], 'cardinality', i["stats"][1])
            statistics.set_value(i["colname"], 'unique_entries', i["stats"][2])
            statistics.set_value(i["colname"], 'coverage_of_uniques', i["stats"][3])

            if i["identifier"]:
                logging.info("Dropping 1st class identifier: {0}".format(i["colname"]))
                del df[i["colname"]] # treatment of 1nd class identifier
            else:
                k_means_input.append([ float(i["stats"][0]), float(i["stats"][1]) ])

        #pool.close()

    # cleanup df
    df = df.dropna(axis=1, how='all')
    colnames = list(df.columns.values)

    if len(colnames) <2:
        logging.info("Stop: There is only 1 col left, no need for 2nd class identifier detection")
        return({"combis_count": 0, "number_of_ucc":0, "iterations_run": 0})

    iterations_run = 1
    list_of_lens = {}

    # using ids for combinations instead of strings due to size
    for index,colname in enumerate(colnames):
        colname_id_mapping[index] = colname

    clustered_columns = cluster_attributes(numpy.asarray(k_means_input),len(colnames),colname_id_mapping)

    # finding 2nd class identifier through out the all combinations of all lengths
    for L in range(2, len(colnames)+1):
        logging.info("{0} / {1}".format(L,len(colnames)+1))

        combinations_for_evaluation = get_combinations_for_evaluation(clustered_columns, colnames, k_means_input, L)

        found_less_than_one = 0
        list_of_lens[L] = []
        if len(quasi_identifier_combinations)==0:
            with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
                #for i in pool.imap_unordered(identify_2nd_identifier, itertools.combinations(range(0,len(colnames)), L)):
                for i in pool.imap_unordered(identify_2nd_identifier, combinations_for_evaluation):
                    if i is not False:
                        if i is not None:
                            quasi_identifier_combinations.append(i)
                            list_of_lens[L].append(len(i))
                        found_less_than_one += 1
                pool.close()
        else:
            with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
                #for i in pool.imap_unordered(is_covered_by_minimal_identifier, itertools.combinations(range(0,len(colnames)), L)):
                for i in pool.imap_unordered(is_covered_by_minimal_identifier, combinations_for_evaluation):
                    if i is not False:
                        if i is not None:
                            quasi_identifier_combinations.append(i)
                            list_of_lens[L].append(len(i))
                        found_less_than_one += 1
                pool.close()

        if found_less_than_one == 0:
            logging.info("Break, there are non combinations to be evaluated. So all minimal UCCs are found!")
            break

        iterations_run += 1

    if settings['find_2nd_class_identifier_exact'] == 'yes':
        generalize_results_from_sample(quasi_identifier_combinations)

    #resolve mapping
    quasi_identifier_combinations_resolved = []
    with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
        for i in pool.map(resolve_id_mapping_combinations, quasi_identifier_combinations):
            quasi_identifier_combinations_resolved.append(i)

    combis_count = analyze_quasi_identifier_combinations(settings, colname_id_mapping, colnames, clustered_columns, global_iteration, list_of_lens, quasi_identifier_combinations_resolved, statistics)

    return({"combis_count": combis_count, "number_of_ucc":len(quasi_identifier_combinations), "iterations_run": iterations_run})


############################## Helper ##############################

# Input a pandas seriesf
def get_entropy(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy
    # scipy.stats.entropy calculates the entropy of a distribution for given probability values.
    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0)
    return entropy


############################## Load Data ##############################

def retrieve_data():
    global df

    df = pd.read_csv(settings['data_path']+settings['source_data'],
                    compression='gzip', header=0, sep=',', quotechar='"',
                    nrows=settings['max_rows'])

    # remove specific attributes from df
    #colnames = list(df.columns.values)

    # or just use the following ones
    #colnames = [
    #                "GivenName","Surname","Gender","NameSet","Title","StreetAddress","City","State","StateFull","ZipCode",
    #                "EmailAddress","Username","Password","TelephoneNumber","Birthday","Age","Color","Occupation","Company",
    #                "BloodType","Kilograms","Centimeters","id","disease_0","disease_date_0","disease_1","disease_date_1",
    #                "disease_2","disease_date_2","disease_3","disease_date_3","disease_4","disease_date_4","disease_5",
    #                "disease_date_5","disease_6","disease_date_6","disease_7","disease_date_7","disease_8","disease_date_8",
    #                "disease_9","disease_date_9","drug_0","drug_date_0","drug_1","drug_date_1","drug_2","drug_date_2",
    #                "drug_3","drug_date_3","drug_4","drug_date_4","drug_5","drug_date_5","drug_6","drug_date_6","drug_7",
    #                "drug_date_7","drug_8","drug_date_8","drug_9","drug_date_9", "gen_0","gen_1","gen_2","gen_3","gen_4","gen_5"
    #                 "Age","Gender","Title","State", "gen_0", "gen_1", "id", "City"
    #            ]
    #data = df[colnames]

    #logging.info("Only using: {0}".format(colnames))
    #return(data)

    return df

############################## Main ##############################
def main():
    logging.basicConfig(filename='log/panda.log', level=logging.INFO)

    #loading csv data
    df = retrieve_data()
    colnames = list(df.columns.values)

    print("Data fetched with {0} columns, processing now mit {1} cores".format(settings["amount_of_columns"],settings['num_cores']))
    start_time = time.time()

    multi_processing_identify_partitioning(df[colnames[:settings["amount_of_columns"]]], 1)
    logging.info("Execution took {0} seconds".format((time.time() - start_time)))



multiprocessing.freeze_support()
main()
