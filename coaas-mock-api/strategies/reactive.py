from strategy import Strategy
from profiler import Profiler
from restapiwrapper import Requester

class Reactive(Strategy):
    def __init__(self, attributes, url, db):
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window)
        self.url = url

    def get_result(self, url = None, json = None):   
        response = ''
        if(url == None): 
            response = self.requester.get_response(self.url)
        else:
            response = self.requester.get_response(url)

        if(len(json) != 0):
            modified_response = {}
            for item in json:
                if(item.attribute in response):
                    modified_response[item.attribute] = response[item.attribute]
            response = modified_response

        # push to profiler
        self.profiler.reactive_push(response)
        return response

    def get_current_profile(self):
        self.profiler.get_details()
