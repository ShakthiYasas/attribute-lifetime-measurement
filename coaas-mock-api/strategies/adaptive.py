import time
import datetime
import threading
import statistics

from lib.event import post_event
from agents.dqnagent import DQNAgent
from lib.fifoqueue import FIFOQueue_2
from agents.simpleagent import SimpleAgent

from strategies.strategy import Strategy
from serviceresolver.serviceselector import ServiceSelector

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from profilers.adaptiveprofiler import AdaptiveProfiler

# Adaptive retrieval strategy
# This strategy would retrieve from the context provider only when the freshness can't be met.
# The algorithm is adaptive becasue the freshness decay gradient adapts based on the current infered lifetime.
# i.e. Steep gradient when lifetime is small and shallower gradient when lifetime is longer.
# However, this does not always refresh for the most expensive SLA either. 
# Adaptive create cache misses and potentially vulanarable to data inaccuracies.
# Therefore, a compromise between the greedy and reactive.

class Adaptive(Strategy):  
    __window_counter = 0
    __learning_counter = 0
    __sla_trend = FIFOQueue_2(10)
    __attribute_access_trend = {}
    __cached_hit_access_trend = {}
    __cached_attribute_access_trend = {}
    __entity_access_trend = FIFOQueue_2(100)
    __request_rate_trend = FIFOQueue_2(1000)

    def __init__(self, db, window, isstatic=True, learncycle = 20): 
        self.__cached = {}
        self.__observed = {}
        self.__evaluated = []
        self.__last_actions = []
        self.__reqs_in_window = 0
        self.__isstatic = isstatic
        self.__moving_window = window
        self.__most_expensive_sla = None
        self.__learning_cycle = learncycle

        self.req_rate_extrapolation = None

        self.service_selector = ServiceSelector(db)
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
            self.__window_counter+=1
            time.sleep(self.__moving_window/1000) 
    
    # Clear function that run on the background
    def clear_expired(self) -> None:
        # Multithread the following 3
        Adaptive.__clear_observed(datetime.datetime.now() - datetime.timedelta(milliseconds=self.__moving_window))
        Adaptive.__clear_cached()
        self.__extrapolate_request_rate()
        
        self.__evaluated.clear()
        self.__request_rate_trend.push((self.__reqs_in_window*1000)/self.__moving_window)
        self.__reqs_in_window = 0
        
        curr_his = self.__sla_trend.getlist()
        avg_freshness = statistics.mean([x[0] for x in curr_his])
        avg_price = statistics.mean([x[1] for x in curr_his])
        avg_pen = statistics.mean([x[2] for x in curr_his])
        self.__sla_trend.push(self.__most_expensive_sla)
        self.__most_expensive_sla = (avg_freshness,avg_price,avg_pen)
    
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
                hit_rate = 0
                if(len(access_list)>0):
                    hit_rate = sum(access_list)/len(access_list)  
                
                access_ratio = len(access_list)/Adaptive.__reqs_in_window 

                if(key in Adaptive.__cached_hit_access_trend):
                    if(curr_attr in key in Adaptive.__cached_hit_access_trend[key]):
                        Adaptive.__cached_hit_access_trend[key][curr_attr].push(hit_rate)
                        Adaptive.__cached_attribute_access_trend[key][curr_attr].push(access_ratio)
                    else:
                        Adaptive.__cached_hit_access_trend[key][curr_attr] = FIFOQueue_2(1000).push(hit_rate)
                        Adaptive.__cached_attribute_access_trend[key][curr_attr] = FIFOQueue_2(1000).push(access_ratio)
                else:
                    Adaptive.__cached_hit_access_trend[key] = {
                        curr_attr : FIFOQueue_2(1000).push(hit_rate)
                    }
                    Adaptive.__cached_attribute_access_trend[key] = {
                        curr_attr : FIFOQueue_2(1000).push(access_ratio)
                    }

                Adaptive.__cached[key][curr_attr].clear()

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.__profiler.get_details()

    # Retrieving context data
    def get_result(self, json = None, fthresh = (0.5,1.0,1.0), session = None) -> dict: 
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
            lifetimes = None
            if(self.__isstatic):
                lifetimes = self.service_registry.get_context_producers(entityid,ent['attributes'])

            if(entityid in self.cache_memory.entityhash):
                # Entity is cached
                # Atleast one of the attributes of the entity is already cached 
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
                    self.__learning_counter += 1
                    # Doesn't cache any item until atleast the mid range is reached
                    if(self.__window_counter > self.trend_ranges[1]):
                        caching_attrs = self.__evalute_attributes_for_caching(entityid,
                                                self.__get_attributes_not_cached(entityid, ent['attributes']))
                        if(caching_attrs):
                            new_context.append((entityid,caching_attrs,lifetimes))

                # Multithread this
                if(len(new_context)>0):
                    self.__refresh_cache_for_entity(new_context)
                if(len(refetching)>0):
                    self.__refresh_cache_for_producers(refetching)
            else:
                # Even the entity is not cached previously
                # So, first retrieving the entity
                output[entityid] = self.__retrieve_entity(ent['attributes'],lifetimes)
                self.__learning_counter += 1
                # Evaluate whether to cache               
                # Doesn't cache any item until atleast the mid range is reached
                if(self.__window_counter > 10):
                    # Run this in the background
                    self.__evaluate_for_caching(entityid, output[entityid])

            output[entityid] = self.cache_memory.get_values_for_entity(entityid, ent['attributes'])
            self.__evaluated.append(entityid)

        if(isinstance(self.selective_cache_agent, DQNAgent) and self.__learning_counter % self.__learning_cycle == 0):
            # Trigger learning for the DQN Agent
            self.trigger_agent_learning()

        return output

    # Get attributes not cached for the entity
    def __get_attributes_not_cached(self, entityid, attributes):
        return list(set(attributes) - set(self.cache_memory.get_attributes_of_entity(entityid)))

    # Evaluate for caching
    def __evalute_attributes_for_caching(self, entityid, attributes:list) -> list:
        # Evaluate the attributes to cache or not
        actions = []
        for att in attributes:
            # Translate the entity,attribute pair to a state
            observation = self.__translate_to_state(entityid,att)
            # Select the action for the state using the RL Agent
            action, est_c_lifetime = self.selective_cache_agent.choose_action(observation)
            self.cache_memory.addcachedlifetime(action, est_c_lifetime)
            if(action != (0,0)):
                self.__last_actions.append(action)
                actions.append(action[1])
        
        # Push to Queue to calculate reward in next epoch
        if(not isinstance(self.selective_cache_agent, SimpleAgent)):
            self.__calculate_reward(list(map(lambda x: (entityid, x), actions)))

        return actions

    def __evaluate_for_caching(self, entityid, attributes:dict):
        # Check if this entity has been evaluated for caching in this window
        # Evaluate if the entity can be cached
        is_caching = False
        updated_attr_dict = []
        random_one = []
        if not entityid in self.__evaluated:
            # Entity hasn't been evaluted in this window before
            for att in attributes:
                observation = self.__translate_to_state(entityid,att)
                action, est_c_lifetime = self.selective_cache_agent.choose_action(observation)
                self.cache_memory.addcachedlifetime(action, est_c_lifetime)
                if(action != (0,0) and action[0] == entityid):
                    updated_attr_dict.append(action[1])
                    is_caching = True
                else:
                    if(action != (0,0)):
                        random_one.append(action)
        
        if(is_caching):
            # Add to cache 
            self.__last_actions = [(entityid, x) for x in updated_attr_dict]
            updated_att_dict = {}
            for att in updated_attr_dict:
                updated_att_dict[att] = attributes[att]
            self.cache_memory.save(entityid,updated_att_dict)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:updated_attr_dict})     
            del self.__observed[entityid]
            del self.__entity_access_trend[entityid]
            # Update the observed list for uncached entities and attributes 
            self.__update_observed(entityid, list(set(updated_attr_dict) - set(attributes)))
        else:
            # Update the observed list for uncached entities and attributes 
            self.__update_observed(entityid, attributes)
        
        # Push to Queue to calculate reward in next epoch
        if(not isinstance(self.selective_cache_agent, SimpleAgent)):
            self.__calculate_reward(list(map(lambda x: (entityid, x), updated_attr_dict)))

        if(random_one):
            self.__cache_entity_attribute_pairs(random_one)
            if(not isinstance(self.selective_cache_agent, SimpleAgent)):
                self.__calculate_reward(random_one)
        
    def __cache_entity_attribute_pairs(self, entityttpairs):
        ent_att = {}
        for entity,att in entityttpairs:
            if(entity in ent_att):
                if(att in ent_att[entity]):
                    ent_att[entity].append(att)
                else:
                    ent_att[entity] = [att]
            else:
                ent_att[entity] = [att]

        for entityid, attlist in ent_att:
            lifetimes = self.service_registry.get_context_producers(entityid,attlist)
            li = [(prodid,value['url']) for prodid, value in lifetimes.items()]
            response = self.service_selector.get_response_for_entity(attlist,li)

            self.cache_memory.save(entityid, response)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:[att]})     
            del self.__observed[entityid]
            del self.__entity_access_trend[entityid]
            # Update the observed list for uncached entities and attributes 
            self.__update_observed(entityid, [att])

    def __update_observed(self, entityid, attributes):
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

    # Translating an observation to a state
    def __translate_to_state(self, entityid, att):
        isobserved = self.__isobserved(entityid, att)
        # Access Rates 
        fea_vec = self.__calculate_access_rates(isobserved, entityid, att)
        # Hit Rates and Expectations
        lifetimes = self.service_registry.get_context_producers(entityid,[att])
        # The above step could be optimzed by using a view (in SQL that updates by a trigger)
        new_feas, avg_latency = self.__calculate_hitrate_features(isobserved, entityid, att, lifetimes)
        fea_vec = fea_vec + new_feas
        # Average Cached Lifetime 
        cached_lt_res = self.__db.read_all_with_limit('attribute-cached-lifetime',{
                    'entity': entityid,
                    'attribute': att
                },10)
        if(cached_lt_res):
            avg_lts = list(map(lambda x: x['c_lifetime'], cached_lt_res))
            fea_vec.append(statistics.mean(avg_lts))
        else:
            fea_vec.append(0)

        # Latency
        fea_vec.append(avg_latency)

        # Average Retriveal Cost
        avg_ret_cost = statistics.mean([values['cost'] for prodid, values in lifetimes.items()])
        fea_vec.append(avg_ret_cost)

        return {
            'entityid': entityid,
            'attribute': att,
            'features': fea_vec
        }

    def __calculate_access_rates(self, isobserved, entityid, att):
        fea_vec = []
        if(isobserved):
            # Actual and Expected Access Rates 
            attribute_trend = self.__cached_attribute_access_trend[entityid][att]
            trend_size = attribute_trend.get_queue_size()

            xi = np.array(range(0,trend_size))
            s = InterpolatedUnivariateSpline(xi, attribute_trend, k=2)
            total_size = trend_size + self.trend_ranges[2]
            extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

            for ran in self.trend_ranges:
                fea_vec.append(attribute_trend.get_last_position(ran))
                fea_vec.append(extrapolation[trend_size-1+ran])
        else:
            # Actual and Expected Access Rates 
            # No actual or expected access rates becasue the item is already cached
            fea_vec = [0,0,0,0,0,0]

        return fea_vec
    
    def __extrapolate_request_rate(self):
        req_rate_trend = self.__request_rate_trend
        trend_size = req_rate_trend.get_queue_size()
        self.rr_trend_size = trend_size

        xi = np.array(range(0,trend_size))
        s = InterpolatedUnivariateSpline(xi, req_rate_trend, k=2)
        total_size = trend_size + self.trend_ranges[2]
        self.req_rate_extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

    def __calculate_hitrate_features(self, isobserved, entityid, att, lifetimes):
        fea_vec = []

        avg_rt = self.service_selector.get_average_responsetime_for_attribute(list(map(lambda x,y: x, lifetimes.items())))

        if(isobserved):
            # No current hit rates to report 
            # Expected Hit Rate (If Not Cached)
            local_avg_lt = []
            if(not self.__isstatic):
                for prodid,lts in lifetimes.items():
                    local_avg_lt.append(self.__profiler.get_mean(str(entityid)+'.'+str(prodid)+'.'+att))
            else:
                for prodid,lts in lifetimes.items():
                    if(att in lts['lifetimes']):
                        local_avg_lt.append(lts['lifetimes'][att])

            avg_life = statistics.mean(local_avg_lt) if(local_avg_lt) else 1
            
            if(avg_life < 0):
                fea_vec.append(0)
                fea_vec.append(1)
            else:
                frt = 1-(avg_rt/avg_life)
                fthr = self.__most_expensive_sla[0]
                delta = avg_life*(fthr-frt)
                for ran in self.trend_ranges:
                    fea_vec.append(0)
                    req_at_point = self.req_rate_extrapolation[self.__request_rate_trend.get_queue_size()-1+ran]
                    if(fthr>frt): 
                        if(delta >= (1/req_at_point)):
                            # It is effective to cache
                            # because multiple requests can be served
                            fea_vec.append(1/(delta*req_at_point))
                        else:
                            # It is not effective to cache
                            # because no more that 1 request could be served
                            fea_vec.append(0)
                    else:
                        # It is not effective to cache
                        # because the item is already expired by the time it arrives
                        fea_vec.append(0)
        else:
            # Current Hit Rate (If Cached)
            hit_trend = self.__cached_hit_access_trend[entityid][att]
            trend_size = hit_trend.get_queue_size()

            xi = np.array(range(0,trend_size))
            s = InterpolatedUnivariateSpline(xi, hit_trend, k=2)
            total_size = trend_size + self.trend_ranges[2]
            extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

            for ran in self.trend_ranges:
                fea_vec.append(hit_trend.get_last_position(ran))
                fea_vec.append(extrapolation[trend_size-1+ran])

        return fea_vec, avg_rt

    def __isobserved(self, entityid, attribute):
        if(entityid in self.__observed and attribute in self.__observed[entityid]['attributes']):
            return True
        return False

    def __calculate_reward(self, actions):   
        # new_state, reward = self.selective_cache_agent.step(action)
        # self.selective_cache_agent.store_transition(self.state, action, reward, new_state)
        # Do the next if agent is not DQN
        # self.selective_cache_agent.reward_history.append(reward)
        pass

    # Get Observed Data
    def get_observed(self):
        return self.__observed

    # Get attribute access trend
    def get_attribute_access_trend(self):
        return self.__attribute_access_trend

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

    # Retrive the current configuration of the most expensive SLA  
    def get_most_expensive_sla(self):
        return self.__most_expensive_sla

    def trigger_agent_learning(self) -> None:
        th = LearningThread(self)
        th.start()
        th.join()

class LearningThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        post_event("need_to_learn")

            