import datetime
from math import trunc
from dateutil import parser

from profiler import Profiler
from strategies.strategy import Strategy
from restapiwrapper import ServiceSelector

# Adaptive retrieval strategy
# This strategy would retrieve from the context provider only when the freshness can't be met.
# The algorithm is adaptive becasue the freshness decay gradient adapts based on the current infered lifetime.
# i.e. Steep gradient when lifetime is small and shallower gradient when lifetime is longer.
# However, this does not always refresh for the most expensive SLA either. 
# Adaptive create cache misses and potentially vulanarable to data inaccuracies.
# Therefore, a compromise between the greedy and reactive.

class Adaptive(Strategy):  
    def __init__(self, db, window):
        self.meta = None
        self.db_instance = db
        self.moving_window = window
        self.observed = {}

        self.service_selector = ServiceSelector(db)
        self.profiler = Profiler(db, self.moving_window, self.__class__.__name__.lower())
    
    # Init_cache initializes the cache memory. 
    def init_cache(self):
        # Set current session to profiler if not set
        if(self.profiler.session == None):
            self.profiler.session = self.session

    # Retrieving context data
    def get_result(self, json = None, session = None) -> dict:               
        query = []
        refetching = []
        new_context = []
        now = datetime.datetime.now()

        # Check freshness of requested attributes
        entityid = json['entityId']

        if(entityid in self.cache_memory.entityhash):
            # Entity is cached
            # Atleast one of the attributes of the entity is already cached 
            for item in json['attributes']:
                # Checking if the specified attributes are in cache
                if(self.cache_memory.get_value_by_key(entityid, item['attribute']) != None):
                    # The context attribute is also cached for the entity
                    # Get the index of the cache slot in which the attribute is cached
                    idx = self.profiler.lookup[str(entityid)+'.'+item['attribute']]
                    # Retreive the recent history of retrievals of the attribute
                    l_f = self.profiler.most_recently_used[idx]
                    last_fecth = self.profiler.last_time if not l_f else l_f[-1][1]
                    # Estimated lifetime of the attribute
                    mean_for_att = self.profiler.mean[idx]
                    # Based on the estimated lifetime, calculate the expiry period until the subsequent retrieval
                    expire_time = mean_for_att * (1 - item['freshness'])
                    time_at_expire = last_fecth + datetime.timedelta(milliseconds=expire_time)
                        
                    if(now > time_at_expire):
                        # If the attribute doesn't meet the freshness level (Cache miss)
                        # add the attribute to the need to refresh list.
                        refetching.append(str(entityid)+'.'+item['attribute'])
                        query.append({
                            'session': session,
                            'entity': entityid,
                            'attribute': item['attribute'], 
                            'isHit': False
                            })
                    else:
                        # Cache Hit (Not required to refresh at the moment)
                        query.append({
                            'session': session, 
                            'entity': entityid,
                            'attribute': item['attribute'], 
                            'isHit': True
                            })
                else:
                    # Although the entity is cached, this particular context attribute is not in cache
                    new_context.append(str(entityid)+'.'+item['attribute'])

            # Update the hit rates 

            # All attributes of the entity if atleast 1 attribute require refreshing or fretching
            if(len(refetching)>0 or len(new_context)>0):
                self.refresh_cache(entityid,new_context)

            time_diff = (now - self.meta['start_time']).total_seconds()*1000
            output = {'step': trunc(time_diff/self.meta['sampling_rate'])}

            # Parse output
            if(len(json) != 0):       
                for item in json:
                    cached_item  = self.cache_memory.get_value_by_key(entityid, item['attribute'])
                    if(cached_item != None):
                        output[item['attribute']] = cached_item

            self.db_instance.insert_many('cache_hits', query)
            
            return output
        else:
            # Atleast the entity is not cached previously
            # Evluate if the entity can be cached
            # If caching, then update all the records 
            # Else update the observed list
            if(entityid in self.observed):
                self.observed[entityid].append(datetime.datetime.now())
            else:
                self.observed[entityid] = [datetime.datetime.now()]
            # Background thread to remove stale observations

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.profiler.get_details()

    # Refreshing selected cached items 
    # Parameters: new_context = list of previously uncached context attributes which requires fetching
    def refresh_cache(self, entityid, new_context = []) -> None:
        # Retrive raw context from provider according to the entity
        response = self.service_selector.get_response_for_entity(entityid)
        
        del response['meta']
        time_diff = (datetime.datetime.now() - self.meta['start_time']).total_seconds()*1000
        modified_response = {
            'step': trunc(time_diff/self.meta['sampling_rate'])
        }

        if(len(attributes)>0):
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

        for att in attributes:
            if(att in response):
                modified_response[att] = response[att]
        response = modified_response

        # Push to profilers
        self.profiler.reactive_push(response)

        # Update cache entries
        self.cache_memory.save(response)