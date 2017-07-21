from anonymizer import *

settings = {
    'source_data': 'extended_user_data.csv.gz',
    #'source_data':'MERGED_HEALTH_CLOUD_DATA.csv.gz',
    'data_path': '../../',
    'max_rows': 100000,#500000, # set 0 for unlimited
    'upper_threshold_supression': 999, # in percent
    'upper_threshold_generalization': 9999, # in percent
    'threshold_id_column': 80, # in percent
    'threshold_cardinality': 30,
    'num_cores': (multiprocessing.cpu_count()-1),
    'amount_of_columns': args.columns,
    'max_chunksize_per_processor': 10000,
    'find_2nd_class_identifier_exact': args.exact,
    'do_treatment': args.treatment,
    'eval_treatment': args.eval_treatment,
    'min_weight_to_be_considered': 7,
    'overall_min_mean_weight': 5,
    'min_mean_weight': 7,
    'min_len_for_min_mean_weight': 4,
    'musts_contain_CRM': True
}

def main():
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S_")
    logging.basicConfig(filename='log/'+str(datetime)+'panda_treatment_eval.log', level=logging.INFO)

    #loading csv data
    df = retrieve_data()
    colnames = list(df.columns.values)
    for i in range(2,78):
        logging.info("############### RUN {0} ###############".format(i))
        logging.info("Data fetched with {0} columns, processing now mit {1} cores".format(settings["amount_of_columns"],settings['num_cores']))
        set_settings(settings)
        start_time = time.time()
        result = multi_processing_identify_partitioning(df[colnames[:i]],i)
        global_timer = (time.time() - start_time)
        logging.info("Execution took {0} seconds".format(global_timer))

multiprocessing.freeze_support()
main()
