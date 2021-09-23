import sys, os
sys.path.append(os.path.abspath(os.path.join('../')))

import traceback
import configparser
from flask import Flask
from datetime import datetime
from constants import seconds_in_day
from flask_restful import Resource, Api
from configuration import CarParkConfiguration
from carpark.carParkFactory import CarParkFactory

from lib.mongoclient import MongoClient
from lib.response import parse_response

app = Flask(__name__)
api = Api(app)
start_time = datetime.now()

config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']

db = MongoClient(default_config['ConnectionString'], default_config['DBName'])

class CarParkContext(Resource):
    # Saving the current session 
    current_session = db.insert_one('simulator-sessions', {'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})

    # Setting up configuration
    configuration = CarParkConfiguration(current_session, sample_size = int(default_config['SampleSize']), 
        standard_deviation = default_config['StandardDeviation'], total_time = float(default_config['TotalTime']), 
        skew = default_config['Skew'], sampling_rate = float(default_config['SamplingRate']), 
        variation=default_config['ValueChange'].split(','), planning_period = default_config['PlanningPeriod'], 
        selected_periods = default_config['SelectedPeriods'], random_noise = True if default_config['RandomNoise'] == 'True' else False,
        noise_percentage = float(default_config['NoisePercentage']), min_occupancy = float(default_config['MinOccupancy']))
    
    carpark_factory = CarParkFactory(configuration)
    carpark = carpark_factory.get_carpark()

    meta = {
        'start_time': str(start_time),
        'sampling_rate': float(default_config['SamplingRate'])
    }

    def get(self):
        try:   
            curr_time = datetime.now()
            time_diff = (curr_time - start_time).total_seconds()*1000
            data = self.carpark.get_current_status(time_diff)
            
            #return data and 200 OK code
            return parse_response(data, meta=self.meta), 200  

        except(Exception):
            print('An error occured : ' + traceback.format_exc())

            # return message and 400 Error code
            return parse_response({'message':'An error occured'}), 400  

api.add_resource(CarParkContext, '/carparks')

if __name__ == '__main__':
    app.run(debug=False, port=5000)