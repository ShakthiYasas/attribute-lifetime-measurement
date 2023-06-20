import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

import time
import random
import traceback
import configparser
from datetime import datetime

from lib.response import parse_response

from flask_restful import Resource
from flask import request

# Global Variables
start_time = datetime.now()
config = configparser.ConfigParser()
config.read('config.ini')
car_config = config['BUILDING']

class BuildingContext(Resource):
    # Setting up configuration
    min_latency = float(car_config['MinLatency'])
    max_latency = float(car_config['MaxLatency'])

    meta = {
        'start_time': str(start_time),
        'sampling_rate': float(car_config['SamplingRate'])
    }

    # GET Endpoint 
    def get(self):
        try:   
            args = request.args
            # Creating the response object
            data = {
                'id': args['id'],
                'location':'-37.82961682420122, 145.04699949155528'
            }

            # Simulating variation of response latencies
            time.sleep(random.uniform(self.min_latency, self.max_latency))
            
            # Return data and 200 OK code
            return parse_response(data, meta=self.meta), 200  

        except(Exception):
            print('An error occured : ' + traceback.format_exc())

            # Return message and 400 Error code
            return parse_response({'message':'An error occured'}), 400  