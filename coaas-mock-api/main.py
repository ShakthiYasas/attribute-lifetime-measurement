import time
import json
import traceback
import configparser
from datetime import datetime
from flask import Flask, request
from cache import Cache
from response import parse_response
from flask_restful import Resource, Api
from strategies.strategyfactory import StrategyFactory
from mongoclient import MongoClient

app = Flask(__name__)
api = Api(app)

config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']
db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

class PlatformMock(Resource):
    strategy = default_config['Strategy'].lower()
    strategy_factory = StrategyFactory(strategy, default_config['Strategy'].lower().split(','), db)
    selected_algo = strategy_factory.selected_algo

    selected_algo.moving_window = int(default_config['MovingWindow'])
    selected_algo.attributes = int(default_config['NoOfAttributes'])
    if(strategy != 'reactive'):
        cache_size = int(default_config['CacheSize'])
        if(cache_size < selected_algo.attributes):
            cache_size = selected_algo.attributes
            print('Initializing cache with '+ str(selected_algo.attributes) + ' slots.')
        
        selected_algo.cache_memory = Cache(cache_size)

    current_session = db.insert_one(strategy+'-sessions', {'strategy': strategy, 'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})

    def post(self):
        try:
            start = time.time()
            json_obj = json.loads(request.json())
            data = self.selected_algo.get_result(default_config['BaseURL'], json_obj)
    
            elapsed_time = time.time() - start
            db.insert_one(self.strategy+'-responses', {'session': self.current_session, 'time': elapsed_time})

            return parse_response(data), 200  # return data and 200 OK code
        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            return parse_response({'message':'An error occured'}), 400  # return message and 400 Error code

api.add_resource(PlatformMock, '/contexts')

if __name__ == '__main__':
    app.run(host='localhost', port=5001)