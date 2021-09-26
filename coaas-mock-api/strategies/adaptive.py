import datetime
from math import trunc
from dateutil import parser
from profiler import Profiler
from restapiwrapper import Requester
from strategies.strategy import Strategy

# Adaptive retrieval strategy
# This strategy would retrieve from the context provider only when the freshness can't be met.
# The algorithm is adaptive becasue the freshness decay gradient adapts based on the current infered lifetime.
# i.e. Steep gradient when lifetime is small and shallower gradient when lifetime is longer.
# However, this does not always refresh for the most expensive SLA either. 
# Adaptive create cache misses and potentially vulanarable to data inaccuracies.
# Therefore, a compromise between the greedy and reactive.

class Adaptive(Strategy):   
    def __init__(self, attributes, url, db, window):
        self.url = url
        self.meta = None
        self.db_insatnce = db
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
        time_diff = (datetime.datetime.now() - self.meta['start_time']).total_seconds()*1000
        response['step'] = trunc(time_diff/self.meta['sampling_rate'])

        # Push to profiler
        self.profiler.reactive_push(response)

        # Save items in cache
        self.cache_memory.save(response)

    # Retrieving context data
    def get_result(self, url = None, json = None, session = None):               
        query = []
        refetching = []
        now = datetime.datetime.now()

        if(len(json) != 0):
            # Check freshness of requested attributes
            for item in json:
                # Checking if the value is in cache
                if(self.cache_memory.get_value_by_key(item['attribute']) != None):
                    idx = self.profiler.lookup[item['attribute']]
                    l_f = self.profiler.most_recently_used[idx]
                    last_fecth = self.profiler.last_time if not l_f else l_f[-1][1]
                    mean_for_att = self.profiler.mean[idx]
                    expire_time = mean_for_att * (1 - item['freshness'])
                    time_at_expire = last_fecth + datetime.timedelta(milliseconds=expire_time)
                    
                    if(now > time_at_expire):
                        # If the attribute doesn't meet the freshness level (Cache miss)
                        # add the attribute to the need to refresh list.
                        refetching.append(item['attribute'])
                        query.append({'session': session, 'attribute': item['attribute'], 'isHit': False})
                    else:
                        # Cache Hit (Not required to refresh at the moment)
                        query.append({'session': session, 'attribute': item['attribute'], 'isHit': True})
                
                # Here, the else is not considered because it is assumed for this expriment that we know 
                # the context attrbutes which will be requested in the period and have proactively cached them
                # atleast once in at the Init_cache phase.
        else:
            # Check freshness of all the attributes in response
            for item in json:
                idx = self.profiler.lookup[item['attribute']]
                l_f = self.profiler.most_recently_used[idx]
                last_fecth = self.profiler.last_time if not l_f else l_f[-1][1]
                mean_for_att = self.profiler.mean[idx]
                expire_time = mean_for_att * (1 - item['freshness'])
                time_at_expire = last_fecth + datetime.timedelta(milliseconds=expire_time)
                
                if(now > time_at_expire):
                    # If the attribute doesn't meet the freshness level (Cache miss)
                    # add the attribute to the need to refresh list.
                    refetching.append(item['attribute'])
                    query.append({'session': session, 'attribute': item['attribute'], 'isHit': False})
                else:
                    # Cache Hit (Not required to refresh at the moment)
                    query.append({'session': session, 'attribute': item['attribute'], 'isHit': True})

        # Refresh cache for those attributes which require refreshing
        self.refresh_cache(refetching)

        time_diff = (now - self.meta['start_time']).total_seconds()*1000
        output = {'step': trunc(time_diff/self.meta['sampling_rate'])}

        # Parse output
        if(len(json) != 0):       
            for item in json:
                cached_item  = self.cache_memory.get_value_by_key(item['attribute'])
                if(cached_item != None):
                    output[item['attribute']] = cached_item

        self.db_insatnce.insert_many('adaptive-hits', query)
        
        return output

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.profiler.get_details()

    # Refreshing selected cached items 
    # Parameters: attributes = list of cached context attributes which requires refreshing
    def refresh_cache(self, attributes) -> None:
        # Retrive raw context from provider
        response = self.requester.get_response(self.url)
        
        del response['meta']
        time_diff = (datetime.datetime.now() - self.meta['start_time']).total_seconds()*1000
        modified_response = {
            'step': trunc(time_diff/self.meta['sampling_rate'])
        }

        for att in attributes:
            if(att in response):
                modified_response[att] = response[att]
        response = modified_response

        # Push to profilers
        self.profiler.reactive_push(response)

        # Update cache entries
        self.cache_memory.save(response)

        #run_in_parallel(self.cache_memory.save(response),self.profiler.reactive_push(response))