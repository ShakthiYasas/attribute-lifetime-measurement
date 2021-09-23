from math import trunc
from dateutil import parser
from datetime import datetime
from profiler import Profiler
from restapiwrapper import Requester
from strategies.strategy import Strategy

# Reactive retrieval strategy
# This strategy would retrieve from the context provider for all the requests.
# Therefore, this is only a redirector. No cache involved.
# Reactive is a benchmark for the other 2 stragies since, the highest freshness can be achieved (Data accuracy).
# However, expecting the lowest response time.

class Reactive(Strategy):
    def __init__(self, attributes, url, db, window):
        self.url = url
        self.meta = None 
        self.moving_window = window

        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())     

    def get_result(self, url = None, json = None, session = None):   
        # Set current session to profiler if not set
        if(self.profiler.session == None):
            self.profiler.session = self.session
        
        # Retrieve context data directly from provider 
        response = ''
        if(url == None): 
            response = self.requester.get_response(self.url)
        else:
            response = self.requester.get_response(url)

        # Calculating the current time step in-relation to the context provider to test data accuracy
        if(self.meta == None):
            self.meta = response['meta']
            self.meta['start_time'] = parser.parse(self.meta['start_time'])
        del response['meta']
        time_diff = (datetime.now() - self.meta['start_time']).total_seconds()*1000

        if(len(json) != 0):
            modified_response = {
                'step': trunc(time_diff/self.meta['sampling_rate'])
            }
            for item in json:
                if(item['attribute'] in response):
                    modified_response[item['attribute']] = response[item['attribute']]
            response = modified_response

        # Push to profiler
        self.profiler.reactive_push(response)
        return response

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.profiler.get_details()
