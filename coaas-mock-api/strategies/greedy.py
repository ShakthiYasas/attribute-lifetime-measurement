from strategies.strategy import Strategy
from profiler import Profiler
from restapiwrapper import Requester
#from lib.util import run_in_parallel
from lib.event import subscribe

class Greedy(Strategy):
    def __init__(self, attributes, url, db):
        print('Initializing Greedy Profile')
        self.url = url
        self.att = attributes
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())       

    def init_cache(self):
        self.profiler.session = self.session
        response = self.requester.get_response(self.url)
        self.profiler.reactive_push(response)
        self.cache_memory.save(response)
        
        self.profiler.auto_cache_refresh_for_greedy(self.att) 
        subscribe("need_to_refresh", self.refresh_cache)

    def get_result(self, url = None, json = None, session = None):       
        response = self.cache_memory.get_values()
        if(len(json) != 0):
            modified_response = {}
            for item in json:
                if(item['attribute'] in response):
                    modified_response[item['attribute']] = response[item['attribute']]
            response = modified_response

        return response

    def get_current_profile(self):
        self.profiler.get_details()

    def refresh_cache(self, attribute) -> None:
        response = self.requester.get_response(self.url)
        fetched = {attribute : response[attribute]}
        self.profiler.reactive_push(fetched)
        self.cache_memory.save(fetched)
        #run_in_parallel(self.cache_memory.save(fetched),self.profiler.reactive_push(fetched))

        
        
        
