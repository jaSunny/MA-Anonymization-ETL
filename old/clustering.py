import numpy
import pandas as pd
import os
import sys
import time
import scipy
import scipy.stats
import scipy.cluster
import itertools, collections
#import logging
#import logging.handlers
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('./')


#################################### k means ####################################
#
# source:
# https://stats.stackexchange.com/questions/9850/how-to-plot-data-output-of-clustering
#

def cluster_attributes(colnames_characteristics,number_of_columns,colname_id_mapping):
    ## determining k through the elbow method
    #plot variance for each value for 'k' between 1,10
    initial = [scipy.cluster.vq.kmeans(colnames_characteristics,i) for i in range(1,number_of_columns)]
    plt.plot([var for (cent,var) in initial])
    #plt.show()
    plt.savefig('output/kmeans_elbow_method.png')

    #logging.info("initial: {0}".format(initial))

    ## Assign your observations to classes, and plot them
    #I reckon index 3 (i.e. 4 clusters) is as good as any so
    cent, var = initial[int(number_of_columns/2)-1]

    #use vq() to get as assignment for each obs.
    assignment,cdist = scipy.cluster.vq.vq(colnames_characteristics,cent)

    plt.scatter(colnames_characteristics[:,0], colnames_characteristics[:,1], c=assignment)
    plt.savefig('output/kmeans_clustering_of_colnames.png')
    plt.clf()

    clustering = pd.DataFrame(columns=('entropy', 'cardinality', 'assignment', 'colname'))
    clustering['entropy'] = colnames_characteristics[:,0]
    clustering['cardinality'] = colnames_characteristics[:,1]
    clustering['assignment'] = assignment

    for index, row in clustering.iterrows():
        clustering['colname'][index] = colname_id_mapping[index]

    #clustering.to_csv("output/clustering.csv", sep=';', quotechar='"')

    return clustering

# this method generates representative samples for a given clusters with a given length
# except for a set of the inner cluster assigment
def choose_samples_from_other_cluster(clustered_columns, items_per_cluster, except_assignment_for_colname):
    unique_assignments = list(clustered_columns['assignment'].unique())
    other_assignments = list(clustered_columns['assignment'].unique())
    own_assignment = clustered_columns['assignment'].get_value(except_assignment_for_colname)
    other_assignments.remove(own_assignment)

    sample = []
    for i in other_assignments:
        colname_set = []
        for n in range(1, items_per_cluster+1):
            if n > len(clustered_columns.loc[clustered_columns['assignment'] == i].index):
                break
            colname_set.append(clustered_columns.loc[clustered_columns['assignment'] == i].index[n-1])

        # only add set if its the correct size, otherwise it may already be included
        if len(colname_set) == n:
            sample.append(set(colname_set))

    return sample

def choose_sample_from_each_cluster(clustered_columns, items_per_cluster):
    unique_assignments = clustered_columns['assignment'].unique()

    sample = []

    #TODO: do I really considere here all possible combinations of representatives?
    # e.g. it may be necessary to considere one set with multiple clusters and different amounts per cluster
    for assignment in unique_assignments:
        # settings['sample_size_per_cluster']
        for i in range(1, items_per_cluster+1): # we select the first n records from a cluster for sampling
            sample.append(clustered_columns.loc[clustered_columns['assignment'] == assignment].index[i-1])

    return sample

def generalize_results_from_sample(quasi_identifier_combinations):
    # TODO: link representative results back to cluster neighbors
    for attribute_combination in quasi_identifier_combinations:
        attribute_combination
