import traceback
import configparser
from flask import Flask
from response import parse_response
from flask_restful import Resource, Api
from strategies.strategyfactory import StrategyFactory

app = Flask(__name__)
api = Api(app)

config = configparser.ConfigParser()
config.read('config.ini')
default_config = config['DEFAULT']

class PlatformMock(Resource):
    strategy_factory = StrategyFactory(default_config['Strategy'].lower())
    selected_algo = strategy_factory.selected_algo

    def get(self):
        try:
            data = self.selected_algo.get_result(default_config['BaseURL'])
            return parse_response(data), 200  # return data and 200 OK code
        except(Exception):
            print('An error occured : ' + traceback.format_exc())
            return parse_response({'message':'An error occured'}), 400  # return message and 400 Error code

api.add_resource(PlatformMock, '/contexts')

if __name__ == '__main__':
    app.run()