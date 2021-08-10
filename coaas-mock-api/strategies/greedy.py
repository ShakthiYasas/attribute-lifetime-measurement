from strategies.strategy import Strategy
from profiler import Profiler
from restapiwrapper import Requester
from lib.util import run_in_parallel
from lib.event import subscribe

class Greedy(Strategy):
    def __init__(self, attributes, url, db):
        print('Initializing Greedy Profile')
        self.url = url
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.session)
        self.profiler.auto_cache_refresh_for_greedy(attributes)
        subscribe("need_to_refresh", self.refresh_cache)
        
    def init_cache(self):
        response = self.requester.get_response(self.url)
        self.cache_memory.save(response)

    def get_result(self, url = None, json = None, session = None):       
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

        
        
        
