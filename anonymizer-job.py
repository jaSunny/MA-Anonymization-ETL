"""
    This file holds the content of the anonymizer job, querying
    anonymization tasks from a redis server and processing those
    tasks by sampling, re-identifying 2nd class identifier candidates
    and treating them if necessary.
    Finally, the sanitized dataset is peristed in the predefined
    HANA table.
    Keep in mind to modify the credentials for the redis server and
    HANA instance.

    Those tasks will be prioritized, which have a small amount of
    columns since of the exponential time complexity for processing
    high dimensional datasets.
    Optimizations for the W[2]-complete search are already implemented
    and automatically applied.
"""

from flask import Flask, request
from flask_restful import reqparse, abort, Resource, Api
import pyhdb
import anonymizer
import sys
import string
import random
import json
import redis
import time
import logging
import pandas as pd
from sqlalchemy import create_engine
import multiprocessing
import math
from functools import partial
import numpy
from tqdm import *

app = Flask(__name__)
api = Api(app)
settings = {}

sys.path.append('./')

# fetch credentials from file
def fetch_credentials():
    with open('credentials.json') as credentials_file:
         return json.load(credentials_file)

settings = fetch_credentials()
settings["num_cores"] = (multiprocessing.cpu_count()-1)
settings["non_CRM_quasi_idenifier_check"] = False

# connect to the HANA instance
hana = pyhdb.connect(
    host=settings["hana"]["host"],
    port=settings["hana"]["port"],
    user=settings["hana"]["username"],
    password=settings["hana"]["password"]
)

engine = create_engine('hana://'+settings["hana"]["username"]+':'+settings["hana"]["password"]+'@'+settings["hana"]["host"]+':'+str(settings["hana"]["port"]))

# connect to the redis server
r = redis.StrictRedis(
    host=settings["redis"]["host"],
    port=settings["redis"]["port"],
    db=0
)

def find_job_smallest_colset():
    """
    evalute all given jobs to find the one with
    the smallest number of columns for processing.

    Given the exp. time complexity, those will be
    prioritized which have the smallest ncols
    """
    smallest_colset_value = None
    smallest_colset_key = ""
    smallest_colset_length = 99999

    # iterate over all tasks and find smallest
    for key in r.scan_iter():
        value = r.get(key).decode("utf-8")
        task = json.loads(value)
        colset_length = len(task["columns"])

        if colset_length < smallest_colset_length:
            smallest_colset_value = task
            smallest_colset_key = key
            smallest_colset_length = colset_length

    return smallest_colset_value

# if the table is already present, drop it!
def drop_table_in_HANA(colnames, table_name):
    """ drop a predefined HANA table """
    cursor = hana.cursor()
    stmnt = 'drop table \"NIKOLAI\".\"'+table_name+'\"'
    print(stmnt)
    try:
        cursor.execute(stmnt)
        hana.commit()
        print("table dropped")
    except:
        print("error in table dropping")


def create_table_in_HANA(colnames, table_name):
    """ create a predefined HANA table """
    cursor = hana.cursor()
    stmnt = 'Create column table \"NIKOLAI\".\"'+table_name+'\" ('
    for colname in colnames:
        stmnt += '\"'+ colname + '\" varchar(255), '

    stmnt = stmnt[:-2] + ')'
    print(stmnt)
    try:
        cursor.execute(stmnt)
        hana.commit()
        print("table created")
    except:
        print("error in table creation")

def store_partial_df(df, table_name):
    """
    partial dfs will be stored in separated
    HANA tables with indexes after their naming
    """
    cursor = hana.cursor()
    pbar = tqdm(total=len(df.index))

    for index, row in df.iterrows():
        pbar.update(1)
        statement = 'INSERT INTO \"NIKOLAI\".\"'+table_name+'\" ('
        for colname in map(str, row.index.tolist()):
            statement += '\"'+ colname + '\",'
        statement = statement[:-1] +') VALUES ('
        #for value in map(str, row.tolist()):
        for value in row.tolist():
            if value != value:
                statement += 'null,'
            elif isinstance(value, int) or isinstance(value, float):
                statement +=  str(value) + ','
            else:
                statement += '\''+ str(value) + '\','

        cursor.execute(statement[:-1] +');')

    pbar.close()
    hana.commit()

def store_dfs_in_HANA(df_filenames,table_name,multiprocessing=False):
    """
    This method stores a pregiven df in a predefined table.
    Either by using multiprocessing for parrallized SQL inserting
    or single processed.

    Even though native processing of batches for storing dfs
    through SQL is much quicker, we use multiprocessing to
    store partial dfs in the table, otherwise we will run
    into memory problems really quick.
    """

    for index,df_filename in enumerate(df_filenames):
        df = pd.read_csv(df_filename, compression='gzip', header=0, sep=',', quotechar='"')
        del df["Unnamed: 0"]
        colnames = list(df.columns.values)
        #REMOVE before flight
        drop_table_in_HANA(colnames, table_name)
        create_table_in_HANA(colnames, table_name)
        number_of_parts = math.ceil(len(df.index)/settings['chunksize'])
        number_of_parts = settings['num_cores']

        if multiprocessing:
            with multiprocessing.Pool(settings['num_cores']) as pool:
                pool.imap_unordered(partial(store_partial_df,table_name=table_name), numpy.array_split(df,number_of_parts))
                pool.close()
                pool.join()
        else:
            store_partial_df(df, table_name)

        logging.info("Finished storing {0} df".format(index))

    # dont forget to close the connestion otherwise we may run into
    # connect issues.
    hana.close()


def runner():
    """
    Main functions orchestrating the major work
    """
    logging.basicConfig(filename='log/anonymizer.log', level=logging.INFO)

    task = find_job_smallest_colset()
    if task is None:
        return False

    df = anonymizer.retrieve_data()
    print("Data fetched with {0} columns, processing now".format(len(task["columns"])))

    df_filenames = anonymizer.identify_treat_1st_2nd_class_identifier(df[task["columns"]], len(task["columns"]), retrieve="df")
    store_dfs_in_HANA(df_filenames, task["table_name"])

    status = r.delete(task["job_id"])
    logging.info("Job finished with status {0}".format(status))
    return True

def main():
    while True:
        successful = runner()
        if not successful:
            time.sleep(30) # delays for 30 seconds

if __name__ == '__main__':
    main()
