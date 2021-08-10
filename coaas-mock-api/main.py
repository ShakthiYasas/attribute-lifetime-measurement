import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import time
import json
import traceback
import configparser
from cache import Cache
from datetime import datetime
import matplotlib.pyplot as plt
from flask import Flask, request
from lib.mongoclient import MongoClient
from flask_restful import Resource, Api
from lib.response import parse_response
from strategies.strategyfactory import StrategyFactory

app = Flask(__name__)
api = Api(app)

config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']
db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

class PlatformMock(Resource):
    strategy = default_config['Strategy'].lower()
    strategy_factory = StrategyFactory(strategy, default_config['Strategy'].lower().split(','), default_config['BaseURL'], db)

    current_session = db.insert_one(strategy+'-sessions', {'strategy': strategy, 'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
    
    selected_algo = strategy_factory.selected_algo
    selected_algo.session = str(current_session)
    selected_algo.moving_window = int(default_config['MovingWindow'])
    selected_algo.attributes = int(default_config['NoOfAttributes'])

    if(strategy != 'reactive'):
        cache_size = int(default_config['CacheSize'])
        if(cache_size < selected_algo.attributes):
            cache_size = selected_algo.attributes
            print('Initializing cache with '+ str(selected_algo.attributes) + ' slots.')
        
        selected_algo.cache_memory = Cache(cache_size)

    def post(self):
        try:
            start = time.time()
            json_obj = json.loads(request.json())
            data = self.selected_algo.get_result(default_config['BaseURL'], json_obj, str(self.current_session))
    
            elapsed_time = time.time() - start
            response = parse_response(data, str(self.current_session), str(self.current_session))
            db.insert_one(self.strategy+'-responses', {'session': str(self.current_session), 'strategy': self.strategy, 'data': response, 'time': elapsed_time})

            return response, 200  # return data and 200 OK code
        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            return parse_response({'message':'An error occured'}), 400  # return message and 400 Error code

    def get(self):
        session = request.args.get('session')

        plt.xlabel('Request')
        plt.ylabel('Response Time')
        
        responses = db.read_all(self.strategy+'-responses', {'session': str(self.current_session) if session == None else session, 'strategy': self.strategy})

        requests = []
        responsetimes = []
        for res in responses:
            requests.append(str(res._id))
            responsetimes.append(res.time)

        plt.plot(requests, responsetimes)  
        filename = str(self.current_session)+'-responsetime.png'
        plt.savefig(filename)

        return {'fileName': filename}, 200 # file saved

api.add_resource(PlatformMock, '/contexts')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)