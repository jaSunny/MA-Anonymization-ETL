"""
    This file holds analytic functionality to investigate
    results of an (anonymized) dataset or parts of its
    processing.

    This can be for instances the fitting of distribution
    for a given column, measuring of (mean) weights as
    cardinality of an attribute, calculating the size
    of an binominal coefficient or assessing allowed
    column combinations for a given blacklist of mpmUCCs.
"""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

sys.path.append('./')

#################################### Analysis ####################################

def measure_weight(column_combination):
    """
    Determining the weight (summed cardinality) of a 2nd class identifier tuple
    """
    return math.ceil(sum(colnames_weight[x] for x in column_combination))

def measure_mean_weight(column_combination):
    """
    Determining the mean eight (mean cardinality) of a 2nd class identifier tuple
    """
    column_selection = [colnames_weight[x] for x in column_combination]
    if not column_selection:
        return 0
    estimated_mean = numpy.mean(column_selection)
    return math.ceil(estimated_mean)

def find_distribution(x,y,colname):
    """
    Iterating over all available distributions and determining the
    one with the best fit for a given column / attribute
    """
    h = plt.hist(y, bins=range(48), color='w')
    dist_names = [ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford',
                    'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull',
                    'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                    'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l',
                    'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper',
                    'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz',
                    'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm',
                    'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb',
                    'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma',
                    'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami',
                    'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw',
                    'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh',
                    'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon',
                    'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min',
                    'weibull_max', 'wrapcauchy'
                ]
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * len(x)
        plt.plot(pdf_fitted, label=dist_name)
        plt.xlim(0,47)

    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig(colname+'.png')
    plt.clf()

def get_allowed_combinations(colnames, highest_prio_identifiers, quasi_identifier_combinations):
    """
    Assessing allowed column combinations by evaluating
    all identified quasi identifier tuple
    """
    sanitized_attribute_combinations = []
    # create sanitized dfs
    for entry in highest_prio_identifiers:
        evaluated_attribute = entry[0]
        #print("quasi_identifier: {0}".format(evaluated_attribute))

        unique_blacklisted_attributes =  []
        for sublist in quasi_identifier_combinations:
            if evaluated_attribute in sublist:
                for attribute in sublist:
                    if attribute not in unique_blacklisted_attributes:
                        unique_blacklisted_attributes.append(attribute)

        #print("unique_blacklisted_attributes: {0}".format(unique_blacklisted_attributes))

        #allowed_attributes = colnames - unique_blacklisted_attributes
        allowed_attributes = [item for item in colnames if item not in unique_blacklisted_attributes]
        # check if allowed_attribute is empty
        if not allowed_attributes:
            logging.info("No attribute combinations allowed for {0}, therefore, dropping!".format(evaluated_attribute))
        else:
            #print("allowed_attributes: {0}".format(allowed_attributes))
            selected_columns = allowed_attributes
            selected_columns.append(evaluated_attribute)
            #print("selected_columns: {0}".format(selected_columns))
            if selected_columns not in sanitized_attribute_combinations:
                sanitized_attribute_combinations.append(selected_columns)
            else:
                logging.info("{0} in {1}".format(selected_columns, sanitized_attribute_combinations))

    logging.info("sanitized_attribute_combinations: {0}".format(sanitized_attribute_combinations))
    return sanitized_attribute_combinations

def get_attribute_specific_round_statistic(attribute_combinations):
    #return({'ucc_found':len(attribute_combinations), 'mean_len': numpy.mean(attribute_combinations), 'standard_diviation': numpy.std(attribute_combinations)})
    return({'ucc_found':len(attribute_combinations)})

def get_round_statistic(list_of_lens):
    round_statistics = pd.DataFrame(columns=('round_length', 'ucc_found', 'mean_len','standard_diviation'))
    round_statistics = round_statistics.set_index(['round_length'])

    for run, items in list_of_lens.items():
        round_statistics.set_value(run, 'ucc_found', len(items))
        round_statistics.set_value(run, 'mean_len', numpy.mean(items))
        round_statistics.set_value(run, 'standard_diviation', numpy.std(items))

    return round_statistics

def get_highest_prio_identifiers(quasi_identifier_combinations):
    """
    Determining the attribute with the highest priority from all
    identified quasi identifier tuple by counting and sorting the
    present attributes and return the one with the highest appearance
    as count
    """
    highest_prio_identifiers = collections.Counter(itertools.chain(*quasi_identifier_combinations))
    # sort decreasing
    highest_prio_identifiers = [(k, highest_prio_identifiers[k]) for k in sorted(highest_prio_identifiers, key=highest_prio_identifiers.get, reverse=True)]
    return highest_prio_identifiers

############################## Helper ###################################################

def ncr(r):
    """
    Calculating the number of combinations for a given r
    over the sum of all possible length
    """
    n = length_of_colnames
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    result =  numer//denom
    return result

def write_to_file(data, name):
    """
    persisting given data to a file
    """
    with open("output/"+name+".data", "w") as text_file:
        print("{0}".format(data), file=text_file)

############################## MAIN ###################################################

#def analyze_quasi_identifier_combinations(settings, colname_id_mapping, colnames, clustered_columns, global_iteration, list_of_lens, quasi_identifier_combinations, statistics):
def analyze_quasi_identifier_combinations(settings, colname_id_mapping_local, colnames, global_iteration, quasi_identifier_combinations, statistics, colnames_weight_local):
    """
    Method for analyzing the identified quasi identifier tuple in regards
    to the occurance, mean length, allowed combinations, mean allowed combinations
    and several more metrics.
    """
    global colname_id_mapping
    colname_id_mapping = colname_id_mapping_local
    global colnames_weight
    colnames_weight = colnames_weight_local
    global length_of_colnames
    length_of_colnames = len(colnames)+1
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S_")

    highest_prio_identifiers = get_highest_prio_identifiers(quasi_identifier_combinations)
    sanitized_attribute_combinations = get_allowed_combinations(colnames, highest_prio_identifiers, quasi_identifier_combinations)

    for identifier in highest_prio_identifiers:
        uccs_with_identifier = [item for item in quasi_identifier_combinations if identifier[0] in item]
        uccs_with_identifier_lengths = [len(i) for i in uccs_with_identifier]
        sanitized_combinations_with_identifier = [item for item in sanitized_attribute_combinations if identifier[0] in item]
        sanitized_combinations_with_identifier_lengths = [len(i) for i in sanitized_combinations_with_identifier]

        statistics.set_value(identifier[0], 'occurance_in_ucc', identifier[1])
        statistics.set_value(identifier[0], 'mean_of_ucc_length', numpy.mean(uccs_with_identifier_lengths))
        statistics.set_value(identifier[0], 'allowed_combinations', len(sanitized_combinations_with_identifier))
        statistics.set_value(identifier[0], 'mean_of_allowed_combinations_length', numpy.mean(sanitized_combinations_with_identifier_lengths))

        for length in uccs_with_identifier_lengths:
            attribute_specific_round_statistic = get_attribute_specific_round_statistic([item for item in uccs_with_identifier_lengths if item == length])
            for run, items in attribute_specific_round_statistic.items():
                statistics.set_value(identifier[0], str(length)+'_'+run, items)

    combis_count = 0

    with closing(multiprocessing.Pool(settings['num_cores'])) as pool:
        for i in pool.imap_unordered(ncr, range(2, len(colnames)+1)):
            combis_count += i
        pool.close()


    stats = "In {0} colnames there are {1} combinations and {2} uccs".format(len(colnames), combis_count ,len(quasi_identifier_combinations))

    #get_round_statistic(list_of_lens).to_csv("output/"+str(datetime)+"round_statistics.csv", sep=';', quotechar='"')

    write_to_file(quasi_identifier_combinations, str(datetime)+str(global_iteration)+"_quasi_identifier_combinations")
    write_to_file(highest_prio_identifiers, str(datetime)+str(global_iteration)+"_occurance_of_attributes_in_ucc")
    write_to_file(sanitized_attribute_combinations, str(datetime)+str(global_iteration)+"_sanitized_attribute_combinations")

    #find_distribution(statistics['occurance_in_ucc'].tolist(),statistics['cardinality'].tolist(),"cardinality")

    logging.info("{0}".format(stats))

    plt.scatter(statistics['entropy'].tolist(), statistics['occurance_in_ucc'].tolist())
    plt.savefig('images/'+str(datetime)+'entropy_vs_occurances.png')
    plt.clf()

    plt.scatter(statistics['cardinality'].tolist(), statistics['occurance_in_ucc'].tolist())
    plt.savefig('images/'+str(datetime)+'cardinality_vs_occurances.png')
    plt.clf()

    #if clustered_columns is not None:
    #    for index,colname in enumerate(colnames):
    #        own_assignment = clustered_columns['assignment'].get_value(index)
    #        statistics.set_value(colname_id_mapping[index], 'cluster_size', len(clustered_columns.loc[clustered_columns['assignment'] == own_assignment].index.tolist()))
    #        statistics.set_value(colname_id_mapping[index], 'assignment', own_assignment)

    statistics.to_csv("output/"+str(datetime)+"statistics.csv", sep=';', quotechar='"')

    logging.info("-------------------ANALYSIS------------------------")


    uccs_with_identifier_lengths = [len(i) for i in quasi_identifier_combinations]
    uccs_with_identifier_weights = [measure_weight(i) for i in quasi_identifier_combinations]
    uccs_with_identifier_mean_weights = [measure_mean_weight(i) for i in quasi_identifier_combinations]

    if uccs_with_identifier_lengths:
        max_ucc_size = max(uccs_with_identifier_lengths)
        uccs_for_L = {}
        weight_uccs_for_L = {}
        for m in range(max_ucc_size):
            uccs_for_L[m] = [i for i in quasi_identifier_combinations if len(i) == m]
            weight_uccs_for_L[m] = [measure_weight(i) for i in uccs_for_L[m]]
            #logging.info("For size {0} there are {1} combinations and a mean weight of {2}".format(m, uccs_with_identifier_lengths.count(m),measure_mean_weight(weight_uccs_for_L[m])))
            if not weight_uccs_for_L[m]:
                mean_weight = 0
            else:
                mean_weight = math.ceil(numpy.mean(weight_uccs_for_L[m]))
            logging.info("For size {0} there are {1} combinations and a mean weight of {2}".format(m, uccs_with_identifier_lengths.count(m),mean_weight))

    logging.info("------------------")

    if uccs_with_identifier_weights:
        unique_weights = []
        for i in uccs_with_identifier_weights:
            if i not in unique_weights:
                unique_weights.append(i)

        for index, weight in enumerate(unique_weights):
            logging.info("For weight {0} there are {1} combinations".format(weight, uccs_with_identifier_weights.count(weight)))

        logging.info("------------------")
        unique_mean_weights = []
        for i in uccs_with_identifier_mean_weights:
            if i not in unique_mean_weights:
                unique_mean_weights.append(i)

        for index, weight in enumerate(unique_mean_weights):
            logging.info("For mean weight {0} there are {1} combinations".format(weight, uccs_with_identifier_mean_weights.count(weight)))


        return combis_count
