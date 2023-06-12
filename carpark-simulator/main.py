import sys, os

from CarParkContext import CarParkContext
sys.path.append(os.path.abspath(os.path.join('.')))

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

api.add_resource(CarParkContext, '/carparks')

if __name__ == '__main__':
    app.run(debug=False, port=5000)