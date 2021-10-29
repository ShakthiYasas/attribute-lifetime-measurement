import time
import datetime
import threading

from lib.fifoqueue import FIFOQueue_2
from strategies.strategy import Strategy
from serviceresolver.serviceselector import ServiceSelector

from profilers.adaptiveprofiler import AdaptiveProfiler

# Adaptive retrieval strategy
# This strategy would retrieve from the context provider only when the freshness can't be met.
# The algorithm is adaptive becasue the freshness decay gradient adapts based on the current infered lifetime.
# i.e. Steep gradient when lifetime is small and shallower gradient when lifetime is longer.
# However, this does not always refresh for the most expensive SLA either. 
# Adaptive create cache misses and potentially vulanarable to data inaccuracies.
# Therefore, a compromise between the greedy and reactive.

class Adaptive(Strategy):  
    def __init__(self, db, window, isstatic=True): 
        self.__cached = {}
        self.__observed = {}
        self.__evaluated = []
        self.__reqs_in_window = 0
        self.__isstatic = isstatic
        self.__moving_window = window
        self.__most_expensive_sla = None

        self.__entity_access_trend = FIFOQueue_2(100)
        self.__attribute_access_trend = {}
        self.__cached_attribute_access_trend = {}
        self.__request_rate_trend = FIFOQueue_2(1000)

        self.service_selector = ServiceSelector()
        if(not self.__isstatic):
            self.__profiler = AdaptiveProfiler(db, self.__moving_window, self.__class__.__name__.lower())
    
    # Init_cache initializes the cache memory. 
    def init_cache(self):
        # Set current session to profiler if not set
        if(self.__profiler.session == None):
            self.__profiler.session = self.session

        # Initializing background thread clear observations.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            self.clear_expired()
            # Observing the attributes that has not been cached within the window
            time.sleep(self.__moving_window/1000) 
    
    # Clear function that run on the background
    def clear_expired(self) -> None:
        # Multithread the following 2
        Adaptive.__clear_observed(datetime.datetime.now() - datetime.timedelta(milliseconds=self.__moving_window))
        Adaptive.__clear_cached()
        
        self.__evaluated.clear()
        self.__request_rate_trend.push((self.__reqs_in_window*1000)/self.__moving_window)
        self.__reqs_in_window = 0
        self.__most_expensive_sla = (0,1.0,1.0)
    
    @staticmethod
    def __clear_observed(exp_time):
        for key,value in Adaptive.__observed.items():
            if(value['req_ts'][-1] < exp_time):
                # The entire entity hasn't been accessed recently
                del Adaptive.__observed[key]
                del Adaptive.__entity_access_trend[key]
                del Adaptive.__attribute_access_trend[key]
            else:
                for tstamp in value['req_ts']:
                    if(tstamp < exp_time):
                        value['req_ts'].pop(0)
                    else:
                        access_freq = 0
                        if(Adaptive.__reqs_in_window>0):
                            access_freq = len(value['req_ts'])/Adaptive.__reqs_in_window
                        Adaptive.__entity_access_trend.push(access_freq)
                        break
                
                for curr_attr, access_list in value['attributes'].items():
                    for tstamp in access_list:
                        if(tstamp < exp_time):
                            value['attributes'][curr_attr].pop(0)
                        else:
                            access_freq = 0
                            if(Adaptive.__reqs_in_window>0):
                                access_freq = len(value['attributes'][curr_attr])/Adaptive.__reqs_in_window 
                            if(key in Adaptive.__attribute_access_trend):
                                if(curr_attr in key in Adaptive.__attribute_access_trend[key]):
                                    Adaptive.__attribute_access_trend[key][curr_attr].push(access_freq)
                                else:
                                    Adaptive.__attribute_access_trend[key][curr_attr] = FIFOQueue_2(1000).push(access_freq)
                            else:
                                Adaptive.__attribute_access_trend[key] = {
                                    curr_attr : FIFOQueue_2(1000).push(access_freq)
                                }
                                
                            break
    
    @staticmethod
    def __clear_cached():
        for key,attributes in Adaptive.__cached.items():                
            for curr_attr, access_list in attributes.items():
                access_freq = 0
                if(len(access_list)>0):
                    access_freq = sum(access_list)/len(access_list)  

                if(key in Adaptive.__cached_attribute_access_trend):
                    if(curr_attr in key in Adaptive.__cached_attribute_access_trend[key]):
                        Adaptive.__cached_attribute_access_trend[key][curr_attr].push(access_freq)
                    else:
                        Adaptive.__cached_attribute_access_trend[key][curr_attr] = FIFOQueue_2(1000).push(access_freq)
                else:
                    Adaptive.__cached_attribute_access_trend[key] = {
                        curr_attr : FIFOQueue_2(1000).push(access_freq)
                    }
                Adaptive.__cached[key][curr_attr].clear()

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.__profiler.get_details()

    # Retrieving context data
    def get_result(self, json = None, fthresh = (0,1.0,1.0), session = None) -> dict: 
        self.__reqs_in_window+=1
        if(self.__most_expensive_sla == None):
            self.__most_expensive_sla = fthresh
        else:
            if(fthresh != self.__most_expensive_sla):
                if(fthresh[0]>self.__most_expensive_sla[0]):
                    self.__most_expensive_sla = fthresh
                else:
                    if(fthresh[2]>self.__most_expensive_sla[2]):
                        self.__most_expensive_sla[2] = fthresh[2]
                    if(fthresh[1]<self.__most_expensive_sla[1]):
                        self.__most_expensive_sla[1] = fthresh[1]

        refetching = [] # Freshness not met for the value generated by a producer [(entityid, prodid)]
        new_context = [] # Need to fetch the entity with all attributes [(entityid, [attributes])]
        now = datetime.datetime.now()

        output = {}
        for ent in json:
            # Check freshness of requested attributes
            entityid = ent['entityId']
            if(entityid in self.cache_memory.entityhash):
                # Entity is cached
                # Atleast one of the attributes of the entity is already cached 
                lifetimes = None
                if(self.__isstatic):
                    lifetimes = self.service_registry.get_context_producers(entityid,ent['attributes'])
                
                if(not (entityid in self.__cached)):
                    self.__cached[entityid] = {}

                # Refetch from the producer if atleast 1 of it's attributes are not available or not fresh
                if(self.cache_memory.are_all_atts_cached(entityid, ent['attributes'])):
                    # All of the attributes requested are in cache for the entity
                    for att_name in ent['attributes']:
                        # Get all values from the context producers for the attribute in cache
                        att_in_cache = self.cache_memory.get_value_by_key(entityid, att_name)
                        ishit = 1
                        if(not self.__isstatic):
                            for prodid,val,lastret in att_in_cache:
                                # Estimated lifetime of the attribute
                                mean_for_att = self.__profiler.get_mean(str(entityid)+'.'+str(prodid)+'.'+att_name)  
                                extime = mean_for_att * (1 - fthresh[0])
                                time_at_expire = lastret + datetime.timedelta(milisseconds=extime)
                                if(now > time_at_expire):
                                    # If the attribute doesn't meet the freshness level (Cache miss) from the producer
                                    # add the entity and producer to the need to refresh list.
                                    ishit = 0
                                    refetching.append((entityid,prodid,lifetimes[prodid]['url']))
                                    break
                        else:
                            # Checking if any of the attributes are not fresh
                            for prodid,val,lastret in att_in_cache:
                                lt = lifetimes[prodid]['lifetimes'][att_name]
                                if(lt<0):
                                    continue
                                else:
                                    extime = lt * (1 - fthresh)
                                    time_at_expire = lastret + datetime.timedelta(seconds=extime)
                                    if(now > time_at_expire):
                                        # If the attribute doesn't meet the freshness level (Cache miss) from the producer
                                        # add the entity and producer to the need to refresh list.
                                        ishit = 0
                                        refetching.append((entityid,ent['attributes'],prodid,lifetimes[prodid]['url']))
                                        break
                        
                        if(not (att_name in self.__cached[entityid])):
                            self.__cached[entityid][att_name] = [ishit]
                        else:
                            self.__cached[entityid][att_name].append(ishit)
                else:
                    # Atleast one of the attributes requested are not in cache for the entity
                    # Should return the attributes that should be cached
                    cachig_attrs = self.__evalute_attributes_for_caching(entityid,
                                            self.__get_attributes_not_cached(entityid, ent['attributes']))
                    if(cachig_attrs):
                        new_context.append((entityid,cachig_attrs,lifetimes))

                # Multithread this
                if(len(new_context)>0):
                    self.__refresh_cache_for_entity(new_context)
                if(len(refetching)>0):
                    self.__refresh_cache_for_producers(refetching)
            else:
                # Even the entity is not cached previously
                # So, first retrieving the entity
                output[entityid] = self.__retrieve_entity(ent['attributes'],lifetimes)
                # Evaluate whether to cache
                # Run this in the background
                self.__evaluate_for_caching(entityid, output[entityid])

            output[entityid] = self.cache_memory.get_values_for_entity(entityid, ent['attributes'])
                
        return output

    # Get attributes not cached for the entity
    def __get_attributes_not_cached(self, entityid, attributes):
        return list(set(attributes) - set(self.cache_memory.get_attributes_of_entity(entityid)))

    # Evaluate for caching
    def __evalute_attributes_for_caching(self, entityid, attributes:list) -> list:
        # Evaluate the attributes to cache or not
        # And return those which need to be cached
        pass

    def __evaluate_for_caching(self, entityid, attributes:dict):
        # Check if this entity has been evaluated for caching in this window
        # Evaluate if the entity can be cached
        is_caching = False
        updated_attr_dict = {}
        if not entityid in self.__evaluated:
            # Entity hasn't been evaluted in this window before
            is_caching = True
            # if caching, updated_attr_dict with those which will be cached
        
        if(is_caching):
            # Add to cache 
            self.cache_memory.save(entityid,updated_attr_dict)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:updated_attr_dict})
            del self.__observed[entityid]
            del self.__entity_access_trend[entityid]
        else:
            # Update the observed list for uncached entities and attributes 
            now = datetime.datetime.now()
            if(entityid in self.__observed):
                self.__observed[entityid]['req_ts'].append(now)
                for attr,vals in attributes:
                    if(attr in self.__observed[entityid]['attributes']):
                        self.__observed[entityid]['attributes'][attr].append(now)
                    else:
                        self.__observed[entityid]['attributes'][attr] = [now]
            else:
                attrs = {}
                for attr,vals in attributes:
                    attrs[attr] = [now]
                self.__observed[entityid] = {
                    'req_ts': [now],
                    'attributes': attrs
                } 


    # Retrieving context for an entity
    def __retrieve_entity(self, attribute_list: list, metadata: dict) ->  dict:
        # Retrive raw context from provider according to the entity
        return self.service_selector.get_response_for_entity(attribute_list, 
                    list(map(lambda key,value: (key,value['url']), metadata.items())))

    def __refresh_cache_for_entity(self, new_context) -> None:
        for entityid,attribute_list,metadata in new_context:       
            response = self.service_selector.get_response_for_entity(attribute_list, 
                        list(map(lambda key,value: (key,value['url']), metadata.items())))
            # Save items in cache
            self.cache_memory.save(entityid,response)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:response})

    # Refreshing for selected context producers
    def __refresh_cache_for_producers(self, refresh_context) -> None:
        # Retrive raw context from provider according to the provider
        for entityid,attribute_list,prodid,url in refresh_context:
            response = self.service_selector.get_response_for_entity(attribute_list,[(prodid,url)])
            # Save items in cache
            self.cache_memory.save(entityid,response)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:response})
            
