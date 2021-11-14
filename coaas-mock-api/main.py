import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import _thread
import secrets
import traceback
import configparser
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from flask import Flask, request
from flask_restful import Resource, Api

from lib.mongoclient import MongoClient
from lib.response import parse_response
from lib.sqlliteclient import SQLLiteClient

from cache.cachefactory import CacheFactory
from strategies.strategyfactory import StrategyFactory
from configurations.cacheconfig import CacheConfiguration

from agents.agentfactory import AgentFactory

app = Flask(__name__)
api = Api(app)

# Global variables
config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']

# Connecting to a monogo DB instance 
db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

# Selected Strategy
selected_algo = None

class PlatformMock(Resource):
    # Initialize this session
    __token = secrets.token_hex(nbytes=16)
    strategy = default_config['Strategy'].lower()
    db.insert_one('sessions', 
        {
            'strategy': strategy, 
            'token': __token,
            'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        })
    
    # Create an instance of the refreshing strategy
    __strategy_factory = StrategyFactory(strategy, db, int(default_config['MovingWindow']), 
            True if default_config['IsStaticLife'] == 'True' else False, int(default_config['LearningCycle']))
    
    global selected_algo
    selected_algo = __strategy_factory.get_retrieval_strategy()
    setattr(selected_algo, 'trend_ranges', [int(default_config['ShortWindow']), int(default_config['MidWindow']), int(default_config['LongWindow'])])
        
    # Set current session token
    setattr(selected_algo, 'session', __token)

    # Initalize the SQLLite Instance 
    __service_registry = SQLLiteClient(default_config['SQLDBName'])
    db_file = Path(default_config['SQLDBName']+'.db')
    if(not db_file.is_file()):
        __service_registry.seed_db_at_start()
    setattr(selected_algo, 'service_registry', __service_registry)

    # "Reactive" strategy do not need a cache memory. 
    # Therefore, skipping cache initializing for "Reactive".
    isCaching = default_config['IsCaching'] 
    if(strategy != 'reactive' or isCaching != 'True'):
        #Initializing cache memory
        cache_fac = CacheFactory(CacheConfiguration(default_config), __service_registry)
        setattr(selected_algo, 'cache_memory', cache_fac.get_cache_memory(db))

        # Initialize the Selective Caching Agent
        agent_fac = AgentFactory(default_config['RLAgent'], config, selected_algo)
        setattr(selected_algo, 'selective_cache_agent', agent_fac.get_agent())
    
    selected_algo.init_cache()

    # POST /contexts endpoint
    # Retrives context data. 
    def post(self):
        try:
            start = datetime.now()
            json_obj = request.get_json()
            req_id = secrets.token_hex(nbytes=8)

            # Simple Authenticator
            consumer = json_obj['requester']
            fthr = self.__service_registry.get_freshness_for_consumer(consumer['id'],consumer['sla'])
            if(fthr == None):
                # Return message and 401 Error code
                return parse_response({'message':'Unauthorized'}), 401  

            # Start to process the request
            data = selected_algo.get_result(json_obj['query'], fthr, req_id)
            response = parse_response(data, self.__token)
            _thread.start_new_thread(self.__save_rp_stats, (self.__token,response,start))
                
            # Return data and 200 OK code
            return response, 200    
                
        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            # Return message and 400 Error code
            return parse_response({'message':'An error occured'}), 400 

    def __save_rp_stats(self, token,response,start):
        # Statistics
        db.insert_one('responses-history', 
            {
                'session': token, 
                'data': str(response), 
                'response_time': (datetime.now() - start).total_seconds()
            })

class Statistics(Resource):
    # GET /statistics endpoint.
    # Retrives the details (metadata & statistics) of the current session in progress. 
    def get(self, name):
        global selected_algo
        if(name == 'caches'):
            ent_id = int(request.args.get('id'))
            data = selected_algo.get_cache_statistics(ent_id)
            return data, 200   
        elif(name == 'returns'):
            is_curr = str(request.args.get('current')).lower()
            if(is_curr == 'true'):
                data = selected_algo.get_current_cost()
                return {
                    'cost': data
                }, 200   
            else:
                session = str(request.args.get('session'))
                data = selected_algo.get_cost_variation(session)
                return {
                    'variation': data
                }, 200   
        else:
            return {'message': 'Invalid URL'}, 404   

api.add_resource(PlatformMock, '/contexts')
api.add_resource(Statistics, '/statistics/<string:name>')

if __name__ == '__main__':
    app.run(debug=False, port=5001)