from strategy import Strategy
from profiler import Profiler
from restapiwrapper import Requester
from util import run_in_parallel
from event import subscribe

class Greedy(Strategy):
    def __init__(self, attributes, url, db):
        self.url = url
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window)
        subscribe("need_to_refresh", self.refresh_cache)
        
        response = self.requester.get_response(url)
        self.cache_memory.save(response)

    def get_result(self, url = None, json = None):       
        response = self.cache_memory.get_values()
        if(len(json) != 0):
            modified_response = {}
            for item in json:
                if(item.attribute in response):
                    modified_response[item.attribute] = response[item.attribute]
            response = modified_response

        return response

    def get_current_profile(self):
        self.profiler.get_details()

    def refresh_cache(self, attribute) -> None:
        response = self.requester.get_response(self.url)
        fetched = {attribute : response[attribute]}
        run_in_parallel(self.cache_memory.save(fetched),self.profiler.reactive_push(fetched))

        
        
        
