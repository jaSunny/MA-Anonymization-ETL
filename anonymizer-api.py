"""
    This file holds the anonymizer API service
    which exposes an RESTful API for inquering
    anonymization tasks of a user.
"""
from flask import Flask, request
from flask_restful import reqparse, abort, Resource, Api
import anonymizer
import sys
import string
import random
import json
import redis

app = Flask(__name__)
api = Api(app)
settings = {}

sys.path.append('./')

# fetch credentials from file
def fetch_credentials():
    with open('credentials.json') as credentials_file:
         return json.load(credentials_file)

def abort_if_job_doesnt_exist(job_id):
    if r.get(job_id) is None:
        abort(404, message="Job {} doesn't exist".format(job_id))

def abort_if_job_arguments_missing(args):
    if "partner_id" not in args:
        abort(404, message="partner_id specs are missing")
    if "columns" not in args:
        abort(404, message="columns specs are missing")

# generates (user) IDs
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

settings = fetch_credentials()

# connect to the redis server
r = redis.StrictRedis(
    host=settings["redis"]["host"],
    port=settings["redis"]["port"],
    db=0
)

parser = reqparse.RequestParser()
parser.add_argument('columns', type = list)
parser.add_argument('partner_id', type = int)

# Job
# shows a single jobs item and lets you delete a todo item
class Job(Resource):
    def get(self, job_id):
        abort_if_job_doesnt_exist(job_id)
        return r.get(job_id)

    def delete(self, job_id):
        abort_if_job_doesnt_exist(job_id)
        return r.delete(job_id), 204

    def put(self, job_id):
        args = parser.parse_args()
        abort_if_job_doesnt_exist(job_id)
        abort_if_job_arguments_missing(args)

        # since reqparse does not know how to handle JSON properly:
        # https://stackoverflow.com/questions/19384526/how-to-parse-the-post-argument-to-a-rest-service
        requested_columns = request.json['columns']
        value = r.get(key).decode("utf-8")
        task = json.loads(value)
        task['columns'] = requested_columns
        r.set(job_id, json.dumps(task))
        return task, 201

# JobList
# shows a list of all jobs, and lets you POST to add new tasks
class JobList(Resource):
    def get(self):
        result = []
        for key in r.scan_iter():
            value = r.get(key).decode("utf-8")
            result.append(value)
        return result

    def post(self):
        print("recieved request")
        args = parser.parse_args()
        abort_if_job_arguments_missing(args)
        job_id = str(args['partner_id']) +"_"+id_generator()+"_"+id_generator()
        table_name = str(args['partner_id']) +"_"+id_generator()
        # since reqparse does not know how to handle JSON properly:
        # https://stackoverflow.com/questions/19384526/how-to-parse-the-post-argument-to-a-rest-service
        requested_columns = request.json['columns']
        task = {'job_id':job_id, 'columns': requested_columns, 'partner_id':args['partner_id'], 'table_name':table_name, 'status':'open'}

        df = anonymizer.retrieve_meta_data()
        colnames = list(df.columns.values)

        for req_colname in requested_columns:
            if req_colname not in colnames:
                error_message = "{0} is not a valid attribute".format(req_colname)
                task['status'] = 'failed'
                task['cause'] = error_message
                return task, 400

        r.set(job_id, json.dumps(task))
        return task, 201

##
## Actually setup the Api resource routing here
##
api.add_resource(JobList, '/jobs')
api.add_resource(Job, '/jobs/<job_id>')

if __name__ == '__main__':
    app.run(debug=True)
