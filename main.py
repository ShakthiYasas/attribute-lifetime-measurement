import datetime
import traceback
import configparser
from flask import Flask
from constants import seconds_in_day
from flask_restful import Resource, Api
from carpark.carParkFactory import CarParkFactory
from configuration import CarParkConfiguration

app = Flask(__name__)
api = Api(app)
start_time = datetime.datetime.now()

config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']

class CarParkContext(Resource):
    configuration = CarParkConfiguration(sample_size = int(default_config['SampleSize']), standard_deviation = float(default_config['StandardDeviation']), 
        total_time = float(default_config['TotalTime']), skew = float(default_config['Skew']), sampling_rate = float(default_config['SamplingRate']))
    carpark_factory = CarParkFactory(configuration)
    
    def get(self):
        try:
            carpark = self.carpark_factory.get_carpark()
            curr_time = datetime.datetime.now()
            time_diff = curr_time - start_time
            milisecond_diff = (time_diff.days * seconds_in_day + time_diff.seconds)*1000

            data = carpark.get_current_status(milisecond_diff)
            return {'current_availability': data}, 200  # return data and 200 OK code

        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            return {'message': 'An error occured'}, 400  # return message and 400 Error code

api.add_resource(CarParkContext, '/carparks')

if __name__ == '__main__':
    app.run()