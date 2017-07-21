# Anonymization of High-Dimensional Health Data

> using in-memory applications based on SAP HANA for exact anonymization through suppression, generalization, perturbation and compartmentation.

This repository holds the source code of the MA thesis _Anonymization of High-Dimensional Health Data_ as practicality proof-of-concept implementation. Besides implementing the exact identification of quasi-identifier which is W[2]-complete, different treatment methods are implemented. This repo also contains the necessary functionality the test, analyze and evaluate the anonymization results.

## Repo Structure

```
.
├── cache
│   ├── .keep
├── images
│   ├── .keep
├── l_diversity.py
├── log
│   ├── .keep
├── output
│   ├── .keep
├── credentials.json
├── evaluation.py
├── quick_eval_score_dataset.py
├── README.md
├── requirements.txt
├── scores.txt
└── treatment_eval.sh
├── analyzer.py
├── anonymizer-api.py
├── anonymizer-job.py
├── anonymizer.py
```

* The _cache_ directory holds the serialized mpmUCCs combinations (2nd class identifiers). When using this caching functionality, the identification step can be skipped reducing the processing time for multiple identification iterations on the same dataset.
* The _images_ directory contains visualizations of the processing if enabled. This can be the graphical representation of the 2nd class identifier as Graph or the process of creating compartments as maximal cliques in the inverted graph.
* In the _output_ directory are various files which are generated during the anonymization, including the sanitized dataset, statistics and much more.
* The _log_ directory holds log files of the anonymization procedure.



## Dependencies & Setup


### Overall

```
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-numpy python3-scipy python3-matplotlib redis-server
```

Use the newest python version (at least 3.6.2), since in the previous versions there [is a rare bug in the multiprocessing engine](https://docs.python.org/3/whatsnew/changelog.html) leading to a deadlock.

```
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
```

### API service specific

For the API service, kindly provide a local redis-server by installing:

```
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
```

Keep in mind to adjust the settings in ```/etc/redis/redis.conf``` regarding to your needs. Afterwards restart the service through ```sudo service redis-server restart```.

One may start the API service manually via ```python3 anonymizer-api.py``` or use [phusion passenger](https://www.phusionpassenger.com/library/walkthroughs/start/python.html) for running python apps in combination with a [apache2](https://www.phusionpassenger.com/library/config/apache/intro.html) webserver for a production ready setup.

Have a SAP HANA up and running including valid configuration, extensions and user management.

## Configuration

Make sure that an SAP HANA instance is running as well as a local redis server. Modify the credentials for HANA and redis in the ```credentials.json``` file.

For connecting with python to the HANA instance _pyhdb_ or _sqlalchemy_ is necessary. The former can be found at [github.com/SAP/pyhdb](https://github.com/SAP/PyHDB) or simply through ```pip3 install pyhdb```.

For the lib ```sqlalchemy_hana``` in ```/local/lib/python3.5/dist-packages/sqlalchemy_hana/dialect.py``` all instances of ```unicode``` must be replaced with ```str```.

## API Design

Executing an http post request triggers the action of the synchrous process by sampling from the dataset and checking its anonymity once again. The user will receive the resource details including the table, where the data view is available one.

HTTP POST ```http://localhost:5000/jobs```

```json
{
  "nrows":1000,
  "columns":["Age","Gender","Surname"],
  "partner_id":0
}
```

Following a RESTful service API, a GET requests details for a specific resource, DELETE removes the resource and PUT may be used to update settings in the resource.

For instances by using:

* HTTP GET ```http://localhost:5000/jobs/1```
* HTTP PUT ```http://localhost:5000/jobs/1```
* HTTP DELETE ```http://localhost:5000/jobs/1```

## Anonymizer.py

The anonymizer.py file can be called directory, offering the following arguments:

|Argument (short)| Argument (long)|Description|Default|Example|
| --- | --- | --- |--- |--- |
| -ncol | --columns | Amount of columns to be considered | 12 | 78|
| -exact | --exact | Determining 2nd class identifier exact instead of heuristic | no | {yes,no}|
| -treat | --treatment | Apply treatment? | yes | {yes,no}|
| -evaltreat | --eval_treatment | Shall treatment be evaluated? | yes | {yes,no}|
| -treatment | --overwrite_treatment | Overwrite automated treatment which predefined one | no | compartmentation|
| -analyze | --analyze_results | Shall quasi identifier be analyzed? | no | {yes,no}|
| -score | --data_value_score | Shall data value be evaluated? | no | {yes,no}|
| -size | --data_size | Shall data size be logged? | no | {yes,no}|
| -visualize | --visualize | Shall visualize be created? | no | {yes,no}|
| -cache | --cache | Shall the mpmUCCs / 2nd class identifier be cached? | yes | {yes,no}|
