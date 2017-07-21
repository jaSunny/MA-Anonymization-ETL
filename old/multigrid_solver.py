import numpy
import pandas as pd
import os
import sys
import time
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import scipy
import pyamg # multigrid solver
import itertools, collections
import _thread
import logging
import logging.handlers

sys.path.append('./')
#sys.path.append('../')

settings = {
    'source_data': 'extended_user_data.csv.gz',
    'data_path': '../../',
    'max_rows': 500000, # set 0 for unlimited
    'threshold_supression_vs_compartment': 10, # in percent
    'threshold_id_column': 80, # in percent
    'num_cores': (multiprocessing.cpu_count()-1)
}

def smoothing_aggregation(df):
    #A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
    A = scipy.sparse.csr_matrix(df.values)
    ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
    print(ml)                                           # print hierarchy information
    b = numpy.random.rand(A.shape[0])                      # pick a random right hand side
    x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-8
    print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector
    
############################## Helper ##############################

def retrieve_data():
    global df

    dtypes = {'GivenName': str,
             'Gender': str,
             'StreetAddress': str,
             'City': str,
             'gen_0': str,
             'gen_1': str
             }

    df = pd.read_csv(settings['data_path']+settings['source_data'], compression='gzip', header=0, sep=',', quotechar='"',
                    dtype=dtypes,
                    #converters=dtypes, engine='python')
                    nrows=settings['max_rows'])

    #df = pd.read_csv(settings['data_path']+settings['source_data'], compression='gzip', header=0, sep=',', quotechar='"')#, nrows=settings['max_rows'])

    # remove specific attributes from df
    colnames = list(df.columns.values)
    #sanitised_colnames = colnames.remove('ID')

    logging.info("Dataset contains the following colnames: {0}".format(colnames))
    # or just use the following ones

    # or just use the following ones
    colnames = [
    #                "GivenName","Surname","Gender","NameSet","Title","StreetAddress","City","State","StateFull","ZipCode",
    #                "EmailAddress","Username","Password","TelephoneNumber","Birthday","Age","Color","Occupation","Company",
    #                "BloodType","Kilograms","Centimeters","id","disease_0","disease_date_0","disease_1","disease_date_1",
    #                "disease_2","disease_date_2","disease_3","disease_date_3","disease_4","disease_date_4","disease_5",
    #                "disease_date_5","disease_6","disease_date_6","disease_7","disease_date_7","disease_8","disease_date_8",
    #                "disease_9","disease_date_9","drug_0","drug_date_0","drug_1","drug_date_1","drug_2","drug_date_2",
    #                "drug_3","drug_date_3","drug_4","drug_date_4","drug_5","drug_date_5","drug_6","drug_date_6","drug_7",
    #                "drug_date_7","drug_8","drug_date_8","drug_9","drug_date_9", "gen_0","gen_1","gen_2","gen_3","gen_4","gen_5"]
                     "Age","Gender","Title","State", "gen_0", "gen_1", "id", "drug_0","City"]
    data = df[colnames]

    logging.info("Only using: {0}".format(colnames))
    return(data)

############################## Main ##############################
def main():
    logging.basicConfig(filename='panda.log', level=logging.INFO)

    #logging.debug('This message should go to the log file')
    #logging.info('So should this')
    #logging.warning('And this, too')

    #loading csv data
    df = retrieve_data()

    start_time = time.time()
    smoothing_aggregation(df)
    logging.info("Execution took {0} seconds".format((time.time() - start_time)))

main()
