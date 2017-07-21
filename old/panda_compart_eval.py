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
    'sample_size': 20,
    'sample_redundancy': 33 #in %
}

#################################### helper ########################################################

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_combinations_for_evaluation(colnames_cardinality_weighted, colnames_entropy_weighted):
    colnames = colnames_entropy_weighted
    if (settings['find_2nd_class_identifier_exact'] == 'yes') or (len(colnames) < 15):
        logging.info("Determining exact colnames")
        return [colnames]
    else:
        logging.info("Determining approx. 2nd class identifier due to compartments")

        number_of_overhead_columns = math.ceil(settings['sample_size'] / 100 * settings['sample_redundancy'])
        number_of_non_overhead_columns = math.ceil(settings['sample_size'] - number_of_overhead_columns)
        amount_of_samples = math.ceil(len(colnames) / number_of_non_overhead_columns)

        cardinality_clusters = chunks(colnames,amount_of_samples)

        samples = []
        for cluster in cardinality_clusters:
            sanitized = [x for x in colnames if x not in cluster]
            samples.append(cluster + sanitized[:number_of_overhead_columns])

        for i in range(0,math.ceil(len(colnames)/2)):
            random_colnames = random.sample(colnames, min(number_of_non_overhead_columns,len(colnames)-1))
            sanitized = [x for x in colnames[math.ceil(len(colnames)*0.33):] if x not in random_colnames]
            if len(sanitized)-1 < 1:
                continue
            random_colnames += random.sample(sanitized, min(number_of_overhead_columns,len(sanitized)-1))
            samples.append(random_colnames)

        #logging.info("Samples to be evaluated: {0}".format(samples))
        return samples

        #return get_samples_for_colnames(colnames, sample_size, amount_of_samples)

def get_samples_for_colnames(colnames, sample_size, amount_of_samples):
    samples = []
    for i in range(0, amount_of_samples):
        samples.append(numpy.random.choice(colnames, size=sample_size))

    return samples

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
    #representatives = df.groupby(resolve_id_mapping_combinations(subset), sort=False).size().reset_index().rename(columns={0:'count'})
    representatives = df.groupby(subset, sort=False).size().reset_index().rename(columns={0:'count'})
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

def search_2nd_class_identifier(colnames, statistics, first_class_identifiers):
    iterations_run = 1
    list_of_lens = {}

    # using ids for combinations instead of strings due to size
    for index,colname in enumerate(colnames):
        colname_id_mapping[index] = colname

    high_cardinality_columns = statistics.sort_values(by='cardinality', ascending=False).index.values.tolist()
    high_cardinality_columns_without_first_identifiers = [x for x in high_cardinality_columns if x not in first_class_identifiers]

    high_entropy_columns = statistics.sort_values(by='entropy', ascending=False).index.values.tolist()
    high_entropy_columns_without_first_identifiers = [x for x in high_entropy_columns if x not in first_class_identifiers]

    colname_combinations = get_combinations_for_evaluation(high_cardinality_columns_without_first_identifiers, high_entropy_columns_without_first_identifiers)

    for index,colname_combination in enumerate(colname_combinations):
        logging.info("Sample {0} / {1}".format(index+1,len(colname_combinations)))
        # finding 2nd class identifier through out the all combinations of all lengths
        for L in range(2, len(colname_combination)+1):
            logging.info("--> {0} / {1}".format(L,len(colname_combination)))

            #combinations_for_evaluation = itertools.combinations(range(0,len(colname_combination)), L)
            combinations_for_evaluation = itertools.combinations(colname_combination, L)
            count_eval_combinations_no_supersets = 0
            list_of_lens[L] = []

            if len(quasi_identifier_combinations)==0:
                with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
                    #for i in pool.imap_unordered(identify_2nd_identifier, itertools.combinations(range(0,len(colnames)), L)):
                    for i in pool.imap_unordered(identify_2nd_identifier, combinations_for_evaluation):
                        if i is not False:
                            if i is not None:
                                if i not in quasi_identifier_combinations:
                                    quasi_identifier_combinations.append(i)
                                    list_of_lens[L].append(len(i))
                            count_eval_combinations_no_supersets += 1
                    pool.close()
            else:
                with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
                    #for i in pool.imap_unordered(is_covered_by_minimal_identifier, itertools.combinations(range(0,len(colnames)), L)):
                    for i in pool.imap_unordered(is_covered_by_minimal_identifier, combinations_for_evaluation):
                        if i is not False:
                            if i is not None:
                                quasi_identifier_combinations.append(i)
                                list_of_lens[L].append(len(i))
                            count_eval_combinations_no_supersets += 1
                    pool.close()

            if count_eval_combinations_no_supersets == 0:
                logging.info("Break, there are non combinations to be evaluated. So all minimal UCCs are found!")
                break

            iterations_run += 1

    for colname in colnames:
        coverered = False
        for combination in colname_combinations:
            if colname in combination:
                coverered = True
        if colname in first_class_identifiers:
            coverered = True
        if not coverered:
            logging.info("{0} is not covered in any combination".format(colname))

    return(iterations_run)

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

    k_means_input = []
    first_class_identifiers = []

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
                first_class_identifiers.append(i["colname"])
                del df[i["colname"]] # treatment of 1nd class identifier
            else:
                k_means_input.append([ float(i["stats"][0]), float(i["stats"][1]) ])

        pool.close()

    # cleanup df
    df = df.dropna(axis=1, how='all')
    colnames = list(df.columns.values)

    if len(colnames) < 2:
        logging.info("Stop: There is only 1 col left, no need for 2nd class identifier detection")
        return({"combis_count": 0, "number_of_ucc":0, "iterations_run": 0,"identifiers":[]})

    iterations_run = search_2nd_class_identifier(colnames, statistics, first_class_identifiers)

    logging.info("Found {0} mpmUCCs".format(len(quasi_identifier_combinations)))

    #resolve mapping
    #quasi_identifier_combinations_resolved = []
    #with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
    #    for i in pool.map(resolve_id_mapping_combinations, quasi_identifier_combinations):
    #        quasi_identifier_combinations_resolved.append(i)

    global length_of_colnames
    length_of_colnames = len(colnames)+1

    combis_count = 0
    with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
        for i in pool.imap_unordered(ncr, range(2, len(colnames)+1)):
            combis_count += i
        pool.close()

    return({"combis_count": combis_count, "number_of_ucc":len(quasi_identifier_combinations), "iterations_run": iterations_run, "identifiers":quasi_identifier_combinations})


############################## Helper ##############################

def ncr(r):
    n = length_of_colnames
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    result =  numer//denom
    return result

def retrieve_data():
    global df

    df = pd.read_csv(settings['data_path']+settings['source_data'],
                    compression='gzip', header=0, sep=',', quotechar='"',
                    nrows=settings['max_rows'])
    return df

############################## Main ##############################
def main():
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S_")
    logging.basicConfig(filename='log/'+str(datetime)+'panda_compart_eval.log', level=logging.INFO)

    #loading csv data
    df = retrieve_data()

    colnames = list(df.columns.values)
    timing_results = pd.DataFrame(columns=('colcount', 'timer', 'combis_count', 'number_of_ucc', 'iterations_run' ))
    timing_results = timing_results.set_index(['colcount'])

    for i in range(2, 43):#len(colnames)):
        logging.info("-------------------- ROUND {0} --------------------".format(i))
        start_time = time.time()
        settings["amount_of_columns"] = i
        result = multi_processing_identify_partitioning(df[colnames[:i]],i)
        global_timer = (time.time() - start_time)
        logging.info("Execution took {0} seconds".format(global_timer))

        uccs_with_identifier_lengths = [len(i) for i in result["identifiers"]]
        timing_results.set_value(i, 'timer', round(global_timer,2))
        timing_results.set_value(i, 'combis_count', result["combis_count"])
        timing_results.set_value(i, 'number_of_ucc', result["number_of_ucc"])
        timing_results.set_value(i, 'iterations_run', result["iterations_run"])
        
        if uccs_with_identifier_lengths:
            max_ucc_size = max(uccs_with_identifier_lengths)
            for m in range(2, max_ucc_size):
                timing_results.set_value(i, 'ucc_size_'+str(m), uccs_with_identifier_lengths.count(m))
        #timing_results.loc[i] = [i, round(global_timer,2), result["combis_count"], result["number_of_ucc"], result["iterations_run"]]

    timing_results.to_csv("output/"+str(datetime)+"eval_timing_results.csv", sep=';', quotechar='"')


multiprocessing.freeze_support()
main()
