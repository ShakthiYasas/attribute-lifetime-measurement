import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

import time
import traceback
import configparser
from cache import Cache
from datetime import datetime
import matplotlib.pyplot as plt
from flask import Flask, request
from flask_restful import Resource, Api

from lib.mongoclient import MongoClient
from lib.response import parse_response
from strategies.strategyfactory import StrategyFactory

app = Flask(__name__)
api = Api(app)

# Global variables
config = configparser.ConfigParser()
config.read(os.getcwd()+'/coaas-mock-api/config.ini')
default_config = config['DEFAULT']

# Connecting to a monogo DB instance 
db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

class PlatformMock(Resource):
    strategy = default_config['Strategy'].lower()
    current_session = db.insert_one(strategy+'-sessions', {'strategy': strategy, 'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
    
    strategy_factory = StrategyFactory(strategy, default_config['Attributes'].lower().split(','), default_config['BaseURL'], db, int(default_config['MovingWindow']))

    selected_algo = strategy_factory.selected_algo
    setattr(selected_algo, 'session', str(current_session))
    setattr(selected_algo, 'attributes', int(default_config['NoOfAttributes']))

    # "Reactive" strategy do not need a cache memory. 
    # Therefore, skipping cache initializing for "Reactive".
    if(strategy != 'reactive'):
        cache_size = int(default_config['CacheSize'])
        if(cache_size < selected_algo.attributes):
            cache_size = selected_algo.attributes
            print('Initializing cache with '+ str(selected_algo.attributes) + ' slots.')
        
        #Initializing cache memory
        setattr(selected_algo, 'cache_memory', Cache(cache_size))
        selected_algo.init_cache()

    # POST /contexts endpoint
    # Retrives context data. 
    def post(self):
        try:
            # Start to process the request
            start = time.time()
            json_obj = request.get_json()
            data = self.selected_algo.get_result(default_config['BaseURL'], json_obj, str(self.current_session))
            elapsed_time = time.time() - start
            response = parse_response(data, str(self.current_session))
            # End of processing the request
            
            db.insert_one(self.strategy+'-responses', {'session': str(self.current_session), 'strategy': self.strategy, 'data': response, 'time': elapsed_time})
            
            # Return data and 200 OK code
            return response, 200 

        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            
            # Return message and 400 Error code
            return parse_response({'message':'An error occured'}), 400  

    # GET /context endpoint.
    # Retrives the details (metadata & statistics) of the current session in progress. 
    def get(self):
        session = request.args.get('session')
        if(session == None):
            session = str(db.read_last(self.strategy+'-sessions')._id)

        plt.xlabel('Request')
        plt.ylabel('Response Time')
        
        responses = db.read_all(self.strategy+'-responses', {'session': str(self.current_session) if session == None else session, 'strategy': self.strategy})

        requests = []
        responsetimes = []
        for res in responses:
            requests.append(str(res._id))
            responsetimes.append(res.time)

        # Plots the variation of response times
        plt.plot(requests, responsetimes)  
        filename = str(self.current_session)+'-responsetime.png'
        plt.savefig(filename)

        return {'fileName': filename}, 200 # file saved

api.add_resource(PlatformMock, '/contexts')

if __name__ == '__main__':
    app.run(debug=False, port=5001)