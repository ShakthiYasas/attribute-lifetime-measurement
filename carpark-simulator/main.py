import sys, os
from BikeContext import BikeContext
from BikeParkContext import BikeParkContext
from BuildingContext import BuildingContext
from CarContext import CarContext

from CarParkContext import CarParkContext
from JunctionContext import JunctionContext
from ParkContext import ParkContext
from WeatherContext import WeatherContext
sys.path.append(os.path.abspath(os.path.join('.')))

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

api.add_resource(CarParkContext, '/carparks')
api.add_resource(CarContext, '/cars')
api.add_resource(BikeContext, '/bikes')
api.add_resource(WeatherContext, '/weather')
api.add_resource(BikeParkContext, '/bikeparks')
api.add_resource(JunctionContext, '/junctions')
api.add_resource(BuildingContext, '/buildings')
api.add_resource(ParkContext, '/parks')

if __name__ == '__main__':
    app.run(debug=False, port=5000)