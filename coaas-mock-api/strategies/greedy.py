from math import trunc
from dateutil import parser
from profiler import Profiler
from datetime import datetime
from lib.event import subscribe
from restapiwrapper import Requester
from strategies.strategy import Strategy

# Greedy retrieval strategy
# This strategy is greedy because it attempts to minimize the number of retrievals.
# The algorithm also adaptive. Similar to adaptive strategy, this would adapt the gradient of the decay with respect to the known lifetime.
# The prime objective is to make miss rate = 0 and serve all requests through cache by proactive refreshing.
# Greedy is the most vulanarable to data inaccuracies but expected to be the fastest to respond.

class Greedy(Strategy):
    def __init__(self, attributes, url, db, window):
        self.url = url
        self.meta = None
        self.att = attributes
        self.moving_window = window

        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())       

    # Init_cache initializes the cache memory. 
    # This includes the first retrieval.
    def init_cache(self):
        # Set current session to profiler if not set
        if(self.profiler.session == None):
            self.profiler.session = self.session

        # Retrives the first response from context provider  
        response = self.requester.get_response(self.url)

        # Calculating the current time step in-relation to the context provider to test data accuracy
        self.meta = response['meta']
        self.meta['start_time'] = parser.parse(self.meta['start_time'])
        del response['meta']
        time_diff = (datetime.now() - self.meta['start_time']).total_seconds()*1000
        response['step'] = trunc(time_diff/self.meta['sampling_rate'])

        # Push to profiler
        self.profiler.reactive_push(response)

        # Save items in cache
        self.cache_memory.save(response)
        
        # Subscribing to the event of need to refresh
        self.profiler.auto_cache_refresh_for_greedy(self.att) 
        subscribe("need_to_refresh", self.refresh_cache)

    # Retrieving context data entirely from the cache
    def get_result(self, url = None, json = None, session = None):       
        time_diff = (datetime.now() - self.meta['start_time']).total_seconds()*1000
        output = {'step': trunc(time_diff/self.meta['sampling_rate'])}

        # Parse output
        if(len(json) != 0):
            for item in json:
                cached_item  = self.cache_memory.get_value_by_key(item['attribute'])
                if(cached_item != None):
                    output[item['attribute']] = cached_item

        return output

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.profiler.get_details()

    # Refreshing selected cached item on a period basis 
    # Parameters: attribute = cached context attributes which requires refreshing
    def refresh_cache(self, attribute) -> None:
        # Retrive raw context from provider
        response = self.requester.get_response(self.url)
        
        del response['meta']
        time_diff = (datetime.now() - self.meta['start_time']).total_seconds()*1000
        fetched = {
            attribute : response[attribute],
            'step': trunc(time_diff/self.meta['sampling_rate'])
        }

        # Push to profilers
        self.profiler.reactive_push(fetched)

        # Update cache entries
        self.cache_memory.save(fetched)

        #run_in_parallel(self.cache_memory.save(fetched),self.profiler.reactive_push(fetched))

        
        
        
