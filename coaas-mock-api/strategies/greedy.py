from math import trunc
from profiler import Profiler
from datetime import datetime
from lib.event import subscribe
from restapiwrapper import Requester
from strategies.strategy import Strategy

class Greedy(Strategy):
    def __init__(self, attributes, url, db, window):
        print('Initializing Greedy Profile')
        self.url = url
        self.meta = None
        self.att = attributes
        self.moving_window = window
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())       

    def init_cache(self):
        self.profiler.session = self.session
        response = self.requester.get_response(self.url)

        self.meta = response['meta']
        self.meta['start_time'] = datetime.datetime.strptime(self.meta['start_time'])

        del response['meta']
        time_diff = datetime.now() - self.meta['start_time']
        milisecond_diff = (time_diff.days * 86400 + time_diff.seconds)*1000
        response['step'] = trunc(milisecond_diff/self.meta['sampling_rate'])

        self.profiler.reactive_push(response)
        self.cache_memory.save(response)
        
        self.profiler.auto_cache_refresh_for_greedy(self.att) 
        subscribe("need_to_refresh", self.refresh_cache)

    def get_result(self, url = None, json = None, session = None):       
        response = self.cache_memory.get_values()
       
        time_diff = datetime.now() - self.meta['start_time']
        milisecond_diff = (time_diff.days * 86400 + time_diff.seconds)*1000
        
        if(len(json) != 0):
            modified_response = {
                'step': trunc(milisecond_diff/self.meta['sampling_rate'])
            }
            for item in json:
                if(item['attribute'] in response):
                    modified_response[item['attribute']] = response[item['attribute']]
            response = modified_response

        return response

    def get_current_profile(self):
        self.profiler.get_details()

    def refresh_cache(self, attribute) -> None:
        response = self.requester.get_response(self.url)

        time_diff = datetime.now() - self.meta['start_time']
        milisecond_diff = (time_diff.days * 86400 + time_diff.seconds)*1000
        
        fetched = {
            attribute : response[attribute],
            'step': trunc(milisecond_diff/self.meta['sampling_rate'])
        }
        self.profiler.reactive_push(fetched)
        self.cache_memory.save(fetched)
        #run_in_parallel(self.cache_memory.save(fetched),self.profiler.reactive_push(fetched))

        
        
        
