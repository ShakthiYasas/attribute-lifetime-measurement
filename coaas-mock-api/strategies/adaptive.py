import time
import _thread
import datetime
import threading
import statistics
from typing import Tuple

from lib.event import post_event
from lib.event import subscribe
from agents.dqnagent import DQNAgent
from lib.fifoqueue import FIFOQueue_2
from agents.simpleagent import SimpleAgent
from lib.exceptions import NewContextException

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
    # Dictionaries
    __delay_dict = {}
    __decision_history = {}
    __attribute_access_trend = {}
    __cached_hit_access_trend = {}
    __cached_attribute_access_trend = {}

    # Counters
    rr_trend_size = 0
    __window_counter = 0
    __learning_counter = 0

    # Status
    __already_modified = False

    # Queues
    __sla_trend = FIFOQueue_2(10)
    __entity_access_trend = FIFOQueue_2(100)
    __request_rate_trend = FIFOQueue_2(1000)
    __retrieval_cost_trend = FIFOQueue_2(10)

    __observedLock = threading.Lock()
    __decision_history_lock = threading.Lock()

    def __init__(self, db, window, isstatic=True, learncycle = 20): 
        self.__db = db
        self.__cached = {}
        self.__observed = {}
        self.__evaluated = []
        self.__last_actions = []
        self.__reqs_in_window = 0
        self.__isstatic = isstatic
        self.__local_ret_costs = []
        self.__moving_window = window
        self.req_rate_extrapolation = None
        self.__learning_cycle = learncycle
        self.__most_expensive_sla = (0.5,1.0,1.0)

        self.service_selector = ServiceSelector(db)
        if(not self.__isstatic):
            self.__profiler = AdaptiveProfiler(db, self.__moving_window, self.__class__.__name__.lower())

        subscribe("subscribed_actions", self.sub_cache_item)
          
    # Init_cache initializes the cache memory. 
    def init_cache(self):
        if(self.selective_cache_agent == None):
            self.selective_cache_agent = self.selective_agent_factory.get_agent()
            self.selective_cache_agent.start()
        # Set current session to profiler if not set
        if((not self.__isstatic) and self.__profiler and self.__profiler.session == None):
            self.__profiler.session = self.session
        
        setattr(self.cache_memory, 'caller_strategy', self)
        self.__is_simple_agent = isinstance(self.selective_cache_agent, SimpleAgent)

        # Initializing background thread clear observations.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    ###################################################################################
    # Section 01 - Background Thread
    # This section of the code runs a background thread that, 
    # a) clear expired data from the statistical data structures 
    # b) calculate/estimate the statistics for the current window
    ###################################################################################

    def run(self):
        while True:
            self.clear_expired()
            _thread.start_new_thread(self.__save_current_cost, ())
            # Observing the attributes that has not been cached within the window
            self.__window_counter+=1
            time.sleep(self.__moving_window/1000) 
    
    # Clear function that run on the background
    def clear_expired(self) -> None:
        # Multithread the following 3
        if(self.__window_counter > 3):
            self.__extrapolate_request_rate()

        if(self.__window_counter >= self.trend_ranges[1]): 
            exp_time = datetime.datetime.now()-datetime.timedelta(seconds=self.__moving_window/1000)
            self.__clear_observed(exp_time)
            self.__clear_cached()
            if(not self.__already_modified and self.__is_simple_agent):
                _thread.start_new_thread(self.selective_cache_agent.modify_dicount_rate, (False,))
            else:
                self.__already_modified = False
            if(self.__decision_history):
                _thread.start_new_thread(self.__learning_after_action, ())

        self.__evaluated.clear()
        self.__request_rate_trend.push((self.__reqs_in_window*1000)/self.__moving_window)
        self.__retrieval_cost_trend.push(statistics.mean(self.__local_ret_costs) if self.__local_ret_costs else 0)
        self.__local_ret_costs.clear()
        self.__reqs_in_window = 0
        
        curr_his = self.__sla_trend.getlist()
        new_most_exp_sla = (0.5,1.0,1.0)
        if(len(curr_his)>0):
            avg_freshness = statistics.mean([x[0] for x in curr_his])
            avg_price = statistics.mean([x[1] for x in curr_his])
            avg_pen = statistics.mean([x[2] for x in curr_his])
            new_most_exp_sla = (avg_freshness,avg_price,avg_pen)

        self.__sla_trend.push(self.__most_expensive_sla)
        self.__most_expensive_sla = new_most_exp_sla
    
    def __clear_observed(self,exp_time):
        dup_obs = self.__observed.copy()
        __reqs_in_window = self.__reqs_in_window
        for key,value in dup_obs.items():
            if(value['req_ts'][-1] < exp_time):
                # The entire entity hasn't been accessed recently
                del self.__observed[key]
                del self.__attribute_access_trend[key]
            else:
                invalidtss = [num for num in value['req_ts'] if num < exp_time]
                for i in invalidtss:
                    value['req_ts'].pop(0)

                validtss = value['req_ts']
                access_freq = 0
                if(__reqs_in_window>0):
                    access_freq = 1 if len(validtss) > __reqs_in_window else len(validtss)/__reqs_in_window 
                self.__entity_access_trend.push(access_freq)
                
                for curr_attr, access_list in value['attributes'].items():
                    invalidtss = [num for num in access_list if num < exp_time]
                    for i in invalidtss:
                        value['attributes'][curr_attr].pop(0)

                    validtss = value['attributes'][curr_attr]
                    access_freq = 0
                    if(__reqs_in_window>0):
                        access_freq = 1 if len(validtss) > __reqs_in_window else len(validtss)/__reqs_in_window 
                    if(key in self.__attribute_access_trend):
                        if(curr_attr in self.__attribute_access_trend[key]):
                            self.__attribute_access_trend[key][curr_attr].push(access_freq)
                        else:
                            que = FIFOQueue_2(1000)
                            que.push(access_freq)
                            self.__attribute_access_trend[key][curr_attr] = que
                    else:
                        que = FIFOQueue_2(1000)
                        que.push(access_freq)
                        self.__attribute_access_trend[key] = {
                            curr_attr : que
                        }

    def __clear_cached(self):
        __reqs_in_window = self.__reqs_in_window
        for key,attributes in self.__cached.items():                
            for curr_attr, access_list in attributes.items():
                hit_rate = 0
                if(len(access_list)>0):
                    hit_rate = sum(access_list)/len(access_list)  
                
                access_ratio = 0 if(__reqs_in_window == 0) else len(access_list)/__reqs_in_window
                if(len(access_list) > __reqs_in_window):
                    access_ratio = 1

                if(key in self.__cached_hit_access_trend):
                    if(curr_attr in self.__cached_hit_access_trend[key]):
                        self.__cached_hit_access_trend[key][curr_attr].push(hit_rate)
                        self.__cached_attribute_access_trend[key][curr_attr].push(access_ratio)
                    else:
                        que = FIFOQueue_2(1000)
                        que.push(hit_rate)
                        self.__cached_hit_access_trend[key][curr_attr] = que
                        que_1 = FIFOQueue_2(1000)
                        que_1.push(access_ratio)
                        self.__cached_attribute_access_trend[key][curr_attr] = que_1
                else:
                    que = FIFOQueue_2(1000)
                    que.push(hit_rate)
                    self.__cached_hit_access_trend[key] = {
                        curr_attr : que
                    }
                    que_1 = FIFOQueue_2(1000)
                    que_1.push(access_ratio)
                    self.__cached_attribute_access_trend[key] = {
                        curr_attr : que_1
                    }

                self.__cached[key][curr_attr].clear()

    # Executes learning the agent (parameter synchornization)
    def __learning_after_action(self):
        for action, values in self.__decision_history.items():
            if(values[2] == False and values[3] >= self.__window_counter+4):
                self.__decision_history_lock.acquire()
                updated_val = list(values)
                updated_val[2] = True
                self.__decision_history[action] = tuple(updated_val)
                self.__decision_history_lock.release()

                curr_state = self.__translate_to_state(action[0], action[1])
                diff = self.__window_counter - values[3]
                reward, is_cached = self.__calculate_reward(action[0], action[1], values[0], diff)  
                if(not self.__is_simple_agent):
                    self.selective_cache_agent.onThread(self.selective_cache_agent.learn, (values[0], values[1], reward, curr_state))

                self.__decision_history_lock.acquire()
                del self.__decision_history[action]
                self.__decision_history_lock.release()

    ###################################################################################
    # Section 02 - Retrive Context
    # This section of the code runs multiple forground threads to retrive the context
    # as per the request. This, 
    # 1. First checks the hash tables of the cache memory (to verify whether the item 
    #   is in cache)
    # 2.1. If available fecth from cache and checks the freshness
    #   2.1.1 If fresh returns the value 
    #   2.1.2 If not fresh, refreshes the cache memory and returns the refrehsed item
    # 2.2 If not availeble, then select the suitable context providers and retrive.
    #   Then checks whether the item could be cached in the background.
    #   2.2.1 If could be cached, cache and update the statistcs. 
    #   2.2.2 Else, update waits.
    ###################################################################################

    # Retrieving context data
    def get_result(self, json = None, fthresh = (0.5,1.0,1.0), req_id = None) -> dict: 
        return self.__get_result_when_simple(json, fthresh, req_id)

    # Retrive function when the agent is Simple
    def __get_result_when_simple(self, json = None, fthresh = (0.5,1.0,1.0), req_id = None) -> dict: 
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

        now = datetime.datetime.now()

        output = {}
        for ent in json:
            refetching = [] # Freshness not met for the value generated by a producer [(entityid, prodid)]
            new_context = [] # Need to fetch the entity with all attributes [(entityid, [attributes])]

            # Check freshness of requested attributes
            entityid = ent['entityId']
            lifetimes = None
            if(self.__isstatic):
                lifetimes = self.service_registry.get_context_producers(entityid,ent['attributes'])

            if(entityid in self.cache_memory.get_statistics_all()):
                # Entity is cached
                # Atleast one of the attributes of the entity is already cached 
                if(not (entityid in self.__cached)):
                    self.__cached[entityid] = {}

                output[entityid] = {}
                # Refetch from the producer if atleast 1 of it's attributes are not available or not fresh
                if(self.cache_memory.are_all_atts_cached(entityid, ent['attributes'])):
                    self.__learning_counter -= 1
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
                                if(prodid in lifetimes):
                                    lt = lifetimes[prodid]['lifetimes'][att_name]
                                    if(lt<0):
                                        continue
                                    else:
                                        extime = lt * (1 - fthresh[0])
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
                    if(self.__window_counter >= self.trend_ranges[1]+1):
                        caching_attrs = self.__evalute_attributes_for_caching(entityid,
                                                self.__get_attributes_not_cached(entityid, ent['attributes']))
                        if(caching_attrs):
                            new_context.append((entityid,caching_attrs,lifetimes))
                        self.__evaluated.append(entityid)
                    else:
                        self.__update_observed(entityid, ent['attributes'])

                threads = []
                if(len(new_context)>0):
                    en_re_th = threading.Thread(target=self.__refresh_cache_for_entity(new_context))
                    en_re_th.start()
                    threads.append(en_re_th)
                if(len(refetching)>0):
                    at_re_th = threading.Thread(target=self.__refresh_cache_for_producers(refetching))
                    at_re_th.start()
                    threads.append(at_re_th)
                
                # Waiting for all refreshing to complete
                for t in threads:
                    t.join()
                
                val = self.cache_memory.get_values_for_entity(entityid, ent['attributes'])
                
                for att_name,prod_values in val.items():
                    if(prod_values != None):
                        output[entityid][att_name] = [res[1] for res in prod_values]
                    else:
                        # This is when the entity is not even evaluated to be cached
                        res = self.service_selector.get_response_for_entity([att_name], 
                            list(map(lambda k: (k[0],k[1]['url']), lifetimes.items())))   
                        output[entityid][att_name] = [x[1] for x in res[att_name]]
                        # Update observed
                        self.__observedLock.acquire()
                        self.__update_observed(entityid, [att_name])
                        self.__observedLock.release()
            else:
                # Even the entity is not cached previously
                # So, first retrieving the entity
                out = self.__retrieve_entity(ent['attributes'],lifetimes)
                output[entityid] = {}
                for att_name,prod_values in out.items():
                    output[entityid][att_name] = [res[1] for res in prod_values]
                    
                self.__learning_counter += 1
                # Evaluate whether to cache               
                # Doesn't cache any item until atleast the mid range is reached
                #_thread.start_new_thread(self.__evaluate_and_updated_observed_in_thread, (entityid, out))
                self.__evaluate_and_updated_observed_in_thread(entityid, out)

        if(self.__learning_counter % self.__learning_cycle == 0):
            if(isinstance(self.selective_cache_agent, DQNAgent)):
                # Trigger learning for the DQN Agent
                self.trigger_agent_learning()
            elif(isinstance(self.selective_cache_agent, SimpleAgent)):
                # Modify the discount rate
                self.__already_modified = True
                _thread.start_new_thread(self.selective_cache_agent.modify_dicount_rate, ())

        return output

    def __evaluate_and_updated_observed_in_thread(self, entityid, out):
        if(self.__window_counter >= self.trend_ranges[1]+1):
            # Run this in the background
            self.__evaluate_for_caching(entityid, out)
            self.__evaluated.append(entityid)
        else:
            self.__observedLock.acquire()
            self.__update_observed(entityid, out.keys())
            self.__observedLock.release()

    ###################################################################################
    # Section 03 - Evluating to Cache
    # This section performs the actions which are required to estimate the value of 
    # caching a context item. 
    ###################################################################################

    # Evaluate for caching
    def __evalute_attributes_for_caching(self, entityid, attributes:list) -> list:
        # Evaluate the attributes to cache or not
        actions = []
        now = datetime.datetime.now()
        for att in attributes:
            try:
                if(self.__check_delay(entityid, att) or self.__is_spike(entityid, att)):
                    # Translate the entity,attribute pair to a state
                    observation = self.__translate_to_state(entityid,att)
                    # Select the action for the state using the RL Agent
                    action, (est_c_lifetime, est_delay) = None, (None, None)
                    if(self.__is_simple_agent):
                        action, (est_c_lifetime, est_delay) = self.selective_cache_agent.choose_action(observation)
                        if(action != (0,0)):
                            wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                            self.cache_memory.addcachedlifetime(action, wait_time)
                            self.__last_actions.append(action)
                            actions.append(action[1])
                        else:
                            if(entityid in self.__delay_dict):
                                self.__delay_dict[entityid][att] = self.__window_counter + est_delay
                            else:
                                self.__delay_dict[entityid] = { att: self.__window_counter + est_delay }
                        
                        if(action != (0,0) and action[0] == entityid):
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.addcachedlifetime(action, wait_time)
                                self.__last_actions.append(action)
                                actions.append(action[1])
                        else:
                            if(action != (0,0)):
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.addcachedlifetime(action, wait_time)
                                self.__last_actions.append(action)
                                self.__cache_entity_attribute_pairs([action])
                            else:
                                if(entityid in self.__delay_dict):
                                    self.__delay_dict[entityid][att] = self.__window_counter + est_delay
                                else:
                                    self.__delay_dict[entityid] = { att: self.__window_counter + est_delay }
                    else:
                        self.selective_cache_agent.onThread(self.selective_cache_agent.choose_action, (observation,False))                    
            except NewContextException:
                if(entityid in self.__delay_dict):
                    self.__delay_dict[entityid][att] = self.__window_counter + 1
                else:
                    self.__delay_dict[entityid] = { att: self.__window_counter + 1 }

        return actions         

    def __evaluate_for_caching(self, entityid, attributes:dict):
        # Check if this entity has been evaluated for caching in this window
        # Evaluate if the entity can be cached
        random_one = []
        is_caching = False
        updated_attr_dict = []
        now = datetime.datetime.now()
        if(not (entityid in self.__evaluated)):
            # Entity hasn't been evaluted in this window before
            for att in attributes.keys():
                if(self.__check_delay(entityid, att) or self.__is_spike(entityid, att)):
                    try:
                        observation = self.__translate_to_state(entityid,att)
                        
                        action, (est_c_lifetime, est_delay) = None, (None, None)
                        if(self.__is_simple_agent):
                            action, (est_c_lifetime, est_delay) = self.selective_cache_agent.choose_action(observation)
                            
                            if(action != (0,0) and action[0] == entityid):
                                updated_attr_dict.append(action[1])
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.addcachedlifetime(action, wait_time)
                                is_caching = True
                            else:
                                if(action != (0,0)):
                                    random_one.append((action[0], action[1], est_c_lifetime))
                                else:
                                    if(entityid in self.__delay_dict):
                                        self.__delay_dict[entityid][att] = self.__window_counter + est_delay
                                    else:
                                        self.__delay_dict[entityid] = { att: self.__window_counter + est_delay }
                        else:
                            self.selective_cache_agent.onThread(self.selective_cache_agent.choose_action, (observation,False))  
                    except NewContextException:
                        if(entityid in self.__delay_dict):
                            self.__delay_dict[entityid][att] = self.__window_counter + 1
                        else:
                            self.__delay_dict[entityid] = { att: self.__window_counter + 1 }
                     
        if(is_caching):
            # Add to cache 
            self.__last_actions += [(entityid, x) for x in updated_attr_dict]
            updated_dict = {}
            for att in updated_attr_dict:
                updated_dict[att] = attributes[att]
            self.cache_memory.save(entityid,updated_dict)
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:updated_attr_dict})  
            
            for att in updated_attr_dict:
                del self.__attribute_access_trend[entityid][att]
                del self.__observed[entityid][att]
            if(self.__observed[entityid]):
                del self.__observed[entityid]
            # Update the observed list for uncached entities and attributes 
            self.__observedLock.acquire()
            self.__update_observed(entityid, list(set(updated_attr_dict) - set(attributes)))
            self.__observedLock.release()
        else:
            # Update the observed list for uncached entities and attributes 
            if(self.__is_simple_agent):
                self.__observedLock.acquire()
                self.__update_observed(entityid, attributes)
                self.__observedLock.release()

        if(random_one):
            for entity, att, cachelife in random_one:
                
                wait_time = now + datetime.timedelta(seconds=cachelife)
                self.cache_memory.addcachedlifetime(action, wait_time)
                self.__last_actions.append((entity, att))
            self.__cache_entity_attribute_pairs(random_one, True)
    
    # Check if the context attribute has been evaluated to be delayed for caching
    def __check_delay(self,entity,attr):
        if(entity in self.__delay_dict and attr in self.__delay_dict[entity]):
            if(self.__delay_dict[entity][attr] == -1): return False
            if(self.__window_counter <= self.__delay_dict[entity][attr]): return False
            else: 
                del self.__delay_dict[entity][attr]
                if(not self.__delay_dict[entity]):
                    del self.__delay_dict[entity]
                return True
        else: 
            return True
    
    # Check if the context attribute is observed to show a spike in demand
    def __is_spike(self,entity,attr):
        if(self.__isobserved(entity, attr) and entity in self.__attribute_access_trend):
            att_trend = self.__attribute_access_trend[entity][attr].get_last_range(2)
            if(len(att_trend)<2):
                return False
            if((att_trend[0]*2)>=att_trend[1]):
                return True
            return False
        else: return False  

    # Subcriber method to cache an item
    def sub_cache_item(self, parameters):
        entity = parameters[0]
        attribute = parameters[1]
        est_c_lifetime = parameters[2]
        est_delay = parameters[3]
        action = parameters[4]
        observation = parameters[5]

        now = datetime.datetime.now()
        if(action != 0):
            # The item is to be cached
            # Could be a random one or an item evulated to be cached
            wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
            self.cache_memory.addcachedlifetime((entity, attribute), wait_time)
            self.__last_actions.append((entity, attribute))
            self.__cache_entity_attribute_pairs([(entity, attribute)]) 
        else:
            self.__observedLock.acquire()
            if(entity in self.__delay_dict):
                self.__delay_dict[entity][attribute] = self.__window_counter + est_delay
            else:
                self.__delay_dict[entity] = { attribute: self.__window_counter + est_delay }
            self.__update_observed(entity, attribute)
            self.__observedLock.release()

        self.__decision_history[(entity, attribute)] = (observation, action, False, self.__window_counter)

    ###################################################################################
    # Section 04 - Caching
    # This section performs the caching actions by calling the methods in cache memory
    # instance and updating the statistics. 
    ###################################################################################
       
    def __cache_entity_attribute_pairs(self, entityttpairs, is_random=False):
        ent_att = {}
        for entity,att in entityttpairs:
            if(entity in ent_att):
                if(att in ent_att[entity]):
                    ent_att[entity].append(att)
                else:
                    ent_att[entity] = [att]
            else:
                ent_att[entity] = [att]

        for entityid, attlist in ent_att.items():
            lifetimes = self.service_registry.get_context_producers(entityid,attlist)
            li = [(prodid,value['url']) for prodid, value in lifetimes.items()]
            response = self.service_selector.get_response_for_entity(attlist,li)
            self.cache_memory.save(entityid, response)
            # Push to profiler

            for att in attlist:
                if(not self.__isstatic):
                    self.__profiler.reactive_push({entityid:[att]})    
                del self.__attribute_access_trend[entityid][att]
                if(entityid in self.__observed and att in self.__observed[entityid]):
                    del self.__observed[entityid][att]
            
            if(entityid in self.__observed and not self.__observed[entityid]):
                del self.__observed[entityid]

    # Update statistics when an item is transitioned from not cached to cached.
    def __update_observed(self, entityid, attributes):
        now = datetime.datetime.now()
        if(entityid in self.__observed):
            self.__observed[entityid]['req_ts'].append(now)
            for attr in attributes:
                if(attr in self.__observed[entityid]['attributes']):
                    self.__observed[entityid]['attributes'][attr].append(now)
                else:
                    self.__observed[entityid]['attributes'][attr] = [now]
        else:
            attrs = {}
            for attr in attributes:
                attrs[attr] = [now]
            self.__observed[entityid] = {
                'req_ts': [now],
                'attributes': attrs
            } 

    ###################################################################################
    # Section 05 - Transformation (Feature Vector Generation)
    # This section creates feature vectors for observations.
    ###################################################################################

    # Translating an observation to a state
    def __translate_to_state(self, entityid, att):
        isobserved = self.__isobserved(entityid, att)
        # Access Rates [0-5]
        fea_vec = self.__calculate_access_rates(isobserved, entityid, att)
        # Hit Rates and Expectations [6-11]
        lifetimes = self.service_registry.get_context_producers(entityid,[att])
        # The above step could be optimzed by using a view (in SQL that updates by a trigger)
        new_feas, avg_latency = self.__calculate_hitrate_features(isobserved, entityid, att, lifetimes)
        fea_vec = fea_vec + new_feas
        # Average Cached Lifetime [12]
        cached_lt_res = self.__db.read_all_with_limit('attribute-cached-lifetime',{
                    'entity': entityid,
                    'attribute': att
                },10)
        if(cached_lt_res):
            avg_lts = list(map(lambda x: x['c_lifetime'], cached_lt_res))
            if(not self.__is_simple_agent):
                # The following calculation makes the avergae cache lifetime propotional to the long range
                avg_lts = statistics.mean(avg_lts)/(self.trend_ranges[2]*(self.__moving_window/1000))
                fea_vec.append(avg_lts)
            else:
                fea_vec.append(statistics.mean(avg_lts))
        else:
            fea_vec.append(0)

        # Latency [13]
        fea_vec.append(avg_latency)

        # Average Retriveal Cost [14]
        avg_ret_cost = statistics.mean([values['cost'] for values in lifetimes.values()])
        fea_vec.append(avg_ret_cost)

        # Whether cached or not [15]
        #if(self.cache_memory.is_cached(entityid, att)):
        #    fea_vec.append(1)
        #else:
        #    fea_vec.append(0)

        return {
            'entityid': entityid,
            'attribute': att,
            'features': fea_vec
        }

    # Calculates the access rates for the given observation
    def __calculate_access_rates(self, isobserved, entityid, att):
        fea_vec = []
        if(isobserved):
            # Actual and Expected Access Rates 
            attribute_trend = self.__attribute_access_trend[entityid][att]
            trend_size = attribute_trend.get_queue_size()

            xi = np.array(range(0,trend_size if trend_size>=3 else 3))
            yi = attribute_trend.getlist()
            if(trend_size<3):
                diff = 3 - len(yi)
                last_val = yi[-1]
                for i in range(0,diff):
                    yi.append(last_val)

            s = InterpolatedUnivariateSpline(xi, yi, k=2)
            total_size = trend_size + self.trend_ranges[2]
            extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

            for ran in self.trend_ranges:
                fea_vec.append(attribute_trend.get_last_position(ran))
                exp_ar = extrapolation[trend_size-2+ran]
                if(exp_ar < 0):
                    exp_ar = 0
                elif(exp_ar > 1):
                    exp_ar = 1
                fea_vec.append(exp_ar)

        elif(entityid in self.__cached_attribute_access_trend 
            and att in self.__cached_attribute_access_trend[entityid]):
            # The item is rather cached
            attribute_trend = self.__cached_attribute_access_trend[entityid][att]
            trend_size = attribute_trend.get_queue_size()

            xi = np.array(range(0,trend_size if trend_size>=3 else 3))
            yi = attribute_trend.getlist()
            if(trend_size<3):
                diff = 3 - len(yi)
                last_val = yi[-1]
                for i in range(0,diff):
                    yi.append(last_val)

            s = InterpolatedUnivariateSpline(xi, yi, k=2)
            total_size = trend_size + self.trend_ranges[2]
            extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

            for ran in self.trend_ranges:
                fea_vec.append(attribute_trend.get_last_position(ran))
                exp_ar = extrapolation[trend_size-2+ran]
                if(exp_ar < 0):
                    exp_ar = 0
                elif(exp_ar > 1):
                    exp_ar = 1
                fea_vec.append(exp_ar)
                
        else:
            # Actual and Expected Access Rates 
            # No actual or expected access rates becasue the item is already cached
            fea_vec = [0,0,0,0,0,0]

        return fea_vec
    
    # Calculates the hit rates for the given observation
    def __calculate_hitrate_features(self, isobserved, entityid, att, lifetimes):
        fea_vec = []
        avg_rt = self.service_selector.get_average_responsetime_for_attribute(list(map(lambda x: x[0], lifetimes.items())))

        if(isobserved):
            # No current hit rates to report 
            # Expected Hit Rate (If not Cached)
            fea_vec = self.get_projected_hit_rate(entityid, att, avg_rt, lifetimes)
        else:
            # Current Hit Rate (If Cached)
            if(entityid in self.__cached_hit_access_trend and att in self.__cached_hit_access_trend[entityid]):
                hit_trend = self.__cached_hit_access_trend[entityid][att]
                trend_size = hit_trend.get_queue_size()

                xi = np.array(range(0,trend_size if trend_size>=3 else 3))
                yi = hit_trend.getlist()
                if(trend_size<3):
                    diff = 3 - len(yi)
                    last_val = yi[-1]
                    for i in range(0,diff):
                        yi.append(last_val)

                s = InterpolatedUnivariateSpline(xi, yi, k=2)
                total_size = trend_size + self.trend_ranges[2]
                extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

                for ran in self.trend_ranges:
                    fea_vec.append(hit_trend.get_last_position(ran))
                    exp_hr = extrapolation[trend_size-2+ran]
                    if(exp_hr < 0):
                        exp_hr = 0
                    elif(exp_hr > 1):
                        exp_hr = 1
                    fea_vec.append(exp_hr)
            else:
                # The item is not in cache or not observed either.
                # This means the item is very new. So, delay it by 1 window.
                # Expected Hit Rate (If Not Cached)
                raise NewContextException()
                
        return fea_vec, avg_rt

    ###################################################################################
    # Section 06 - Cache Refreshing
    ###################################################################################
    # Execute cache refreshing for entity
    def __refresh_cache_for_entity(self, new_context) -> None:
        for entityid,attribute_list,metadata in new_context:
            response = self.service_selector.get_response_for_entity(attribute_list, 
                        list(map(lambda k: (k[0],k[1]['url']), metadata.items())))
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

    ###################################################################################
    # Section 07 - Helper Methods
    ###################################################################################
    # Extrapolate the 1-D variation
    def __extrapolate_request_rate(self):
        req_rate_trend = self.__request_rate_trend
        trend_size = req_rate_trend.get_queue_size()
        self.rr_trend_size = trend_size

        xi = np.array(range(0,trend_size if trend_size>=3 else 3))
        yi = req_rate_trend.getlist()
        if(trend_size<3):
            diff = 3 - len(yi)
            last_val = yi[-1]
            for i in range(0,diff):
                yi.append(last_val)

        s = InterpolatedUnivariateSpline(xi, yi, k=2)
        total_size = trend_size + self.trend_ranges[2]
        self.req_rate_extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

    # Using extrapolation, get the projected hit rate for the given time ranges
    def get_projected_hit_rate(self, entityid, att, avg_rt, lifetimes):
        fea_vec = []
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
            # Infinite or very long lifetimes
            fea_vec = [0,1,0,1,0,1]
        else:
            frt = 1-(avg_rt/avg_life)
            fthr = self.__most_expensive_sla[0]
            delta = (avg_life*(frt-fthr))/(1-frt)

            attribute_trend = self.__attribute_access_trend[entityid][att]
            trend_size = attribute_trend.get_queue_size()

            xi = np.array(range(0,trend_size if trend_size>=3 else 3))
            yi = attribute_trend.getlist()
            if(trend_size<3):
                diff = 3 - len(yi)
                last_val = yi[-1]
                for i in range(0,diff):
                    yi.append(last_val)

            s = InterpolatedUnivariateSpline(xi, yi, k=2)
            total_size = trend_size + self.trend_ranges[2]
            ar_extrapolation = (s(np.linspace(0, total_size, total_size+1))).tolist()

            rr_trend_size = self.__request_rate_trend.get_queue_size()

            for ran in self.trend_ranges:
                fea_vec.append(0)
                req_at_point = self.req_rate_extrapolation[rr_trend_size-2+ran]*ar_extrapolation[trend_size-2+ran]

                if(fthr<frt and req_at_point > 0): 
                    if(delta >= (1/req_at_point)):
                        # It is effective to cache
                        # because multiple requests can be served
                        exp_hr = (delta*req_at_point)/((delta*req_at_point)+1)
                        fea_vec.append(exp_hr)
                    else:
                        # It is not effective to cache
                        # because no more that 1 request could be served
                        fea_vec.append(0)
                else:
                    # It is not effective to cache
                    # because the item is already expired by the time it arrives
                    fea_vec.append(0)
        
        return fea_vec

    # Check if the entity attribute pair has been observed previously within the window
    def __isobserved(self, entityid, attribute):
        if(entityid in self.__observed and attribute in self.__observed[entityid]['attributes']):
            return True
        return False

     # Get attributes not cached for the entity
    def __get_attributes_not_cached(self, entityid, attributes):
        return list(set(attributes) - set(self.cache_memory.get_attributes_of_entity(entityid)))

    # Calculate the rewards actions taken in the previous window
    def __calculate_reward(self, entityid, att, previous_state, diff):   
        is_cached = False
        past_sla = self.__sla_trend.get_last_range(diff)
        past_ret_costs = self.__retrieval_cost_trend.get_last_range(diff)
        past_request_rate = self.__request_rate_trend.get_last_range(diff)
        
        total_gain = 0
        total_requests = 0

        if(entityid in self.__cached_attribute_access_trend and 
                att in self.__cached_attribute_access_trend[entityid]):
            # This item has been cached in the last decision epoch
            is_cached = True   
            past_hr = self.__cached_hit_access_trend[entityid][att].get_last_range(diff)
            past_ar = self.__cached_attribute_access_trend[entityid][att].get_last_range(diff)

            for idx in range(0,len(past_ar)):
                price = past_hr[idx]*past_sla[idx][1]
                penalty = (1-past_hr[idx])*past_sla[idx][2]
                out = (1/past_request_rate[idx])*past_ar[idx]
                retrieval = (1-past_hr[idx])*past_ret_costs[idx]

                total_requests += out
                total_gain += out*(price - penalty - retrieval)
            
            # This returns the gain or loss of caching an item per request
            return total_gain/total_requests if total_requests>0 else -30, is_cached
        else:
            # This item was not cached
            expected_vals = []

            expected_ar = []
            short_inc = (previous_state[1] - previous_state[0])/self.trend_ranges[0]
            mid_inc = (previous_state[3] - previous_state[1])/(self.trend_ranges[1] - self.trend_ranges[0])
            long_inc = (previous_state[5] - previous_state[3])/(self.trend_ranges[2] - self.trend_ranges[1])
            
            curr_ar = previous_state[1]
            for i in range(0,self.self.trend_ranges[2]):
                if(i < self.trend_ranges[0]):
                    curr_ar += short_inc
                elif(self.trend_ranges[0] <= i < self.trend_ranges[1]):
                    curr_ar += mid_inc
                else:
                    curr_ar += long_inc
                expected_ar.append(curr_ar)

            expected_hr = []
            short_inc = (previous_state[6] - previous_state[7])/self.trend_ranges[0]
            mid_inc = (previous_state[9] - previous_state[7])/(self.trend_ranges[1] - self.trend_ranges[0])
            long_inc = (previous_state[11] - previous_state[9])/(self.trend_ranges[2] - self.trend_ranges[1])

            curr_hr = previous_state[1]
            for i in range(0,self.self.trend_ranges[2]):
                if(i < self.trend_ranges[0]):
                    curr_hr += short_inc
                elif(self.trend_ranges[0] <= i < self.trend_ranges[1]):
                    curr_hr += mid_inc
                else:
                    curr_hr += long_inc
                expected_hr.append(curr_hr)
            
            # Expected Values
            for idx in range(1,len(diff)):
                price = expected_hr[idx]*past_sla[idx][1]  
                penalty = (1-expected_hr[idx])*past_sla[idx][2]
                retrieval = (1-expected_hr[idx])*past_ret_costs[idx]
                out = (1/past_request_rate[idx])*expected_ar[idx]

                total_requests += out
                expected_vals.append(out*(price - penalty - retrieval))

            # Observed Values
            observed_vals = []
            past_ar = self.__attribute_access_trend[entityid][att].get_last_range(diff)
            for idx in range(0,len(past_ar)):            
                penalty = past_sla[idx][2]
                retrieval = past_ret_costs[idx]
                out = (1/past_request_rate[idx])*past_ar[idx]

                total_requests += out
                observed_vals.append(out*(0 - penalty - retrieval))

            diff = sum([observed_vals[i] - expected_vals[i] for i in range(0,diff)])
            
            # This returns the regret of not caching the item
            return diff/total_requests if total_requests>0 else 10, is_cached

    # Retrieving context for an entity
    def __retrieve_entity(self, attribute_list: list, metadata: dict) ->  dict:
        # Retrive raw context from provider according to the entity
        return self.service_selector.get_response_for_entity(attribute_list, 
                    list(map(lambda k: (k[0],k[1]['url']), metadata.items())))

    # Save cost variation
    def __save_current_cost(self):
        self.__db.insert_one('returnofcaching', {
            'session':self.session,
            'window': self.__window_counter - 1,
            'return': self.get_current_cost()
        })

    ###################################################################################
    # Section 08 - Eviction Helpers
    ###################################################################################

    # Reevaluating the nessecity to continue caching an item for the eviction algorithm.
    def reevaluate_for_eviction(self, entityid, att):
        try:
            observation = self.__translate_to_state(entityid,att)
            
            action = None
            if(self.__is_simple_agent):
                action = self.selective_cache_agent.choose_action(observation, skipRandom = True)
            else:
                self.selective_cache_agent.onThread(self.selective_cache_agent.choose_action, (observation, True))

            return action
        except Exception:
            return ((0,0),(0,0))  

    ###################################################################################
    # Section 09 - Getters and Setters
    ###################################################################################

    # Get Observed Data
    def get_observed(self):
        return self.__observed

    # Get attribute access trend
    def get_attribute_access_trend(self):
        return self.__attribute_access_trend

    # Get the current configuration of the most expensive SLA  
    def get_most_expensive_sla(self):
        return self.__most_expensive_sla

    # Set off the learning routine for the reforcement learnign agent
    def trigger_agent_learning(self) -> None:
        th = LearningThread(self)
        th.start()
        th.join()

    # Get a snap shot of the cache memory statistics
    def get_cache_statistics(self, entityid):
        res = self.cache_memory.get_statistics_entity(entityid)
        output = {
                'cached_attributes': [i for i in res.keys()] if res else {}
            }
        if(self.__is_simple_agent):
            output['discount_rate'] = self.selective_cache_agent.get_discount_rate()
        return output

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.__profiler.get_details()

    # Returns the variation of average cost of context caching
    def get_cost_variation(self, session = None):
        output = {}
        hit_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        if(session):
            stats = self.__db.read_all('returnofcaching', {'session': session})
            if(stats):
                for stat in stats:
                    output[stat['window']] = stat['return']
        else:
            if(self.__window_counter >= self.trend_ranges[1]):
                hit_rate = self.cache_memory.get_hitrate_trend().get_last_range(10)

            slas = self.__sla_trend.getlist()
            request_rates = self.__request_rate_trend.get_last_range(10)
            ret_costs = self.__retrieval_cost_trend.get_last_range(10)
            for i in range(0,10):
                idx = -1-i
                try:
                    hr = hit_rate[idx]
                    sla = slas[idx]
                    ret_cost = ret_costs[idx]
                    output[idx] = request_rates[idx]*((hr*sla[1]) - ((1-hr)*sla[2]) - ((1-hr)*ret_cost))
                except Exception:
                    output[idx] = 0

        return output
    
    def get_current_cost(self):
        hit_rate = 0
        if(self.__window_counter >= self.trend_ranges[1]):
            hr = self.cache_memory.get_hitrate_trend().get_last()
            hit_rate = hr[0] if isinstance(hr,Tuple) else hr
        
        sla = self.__sla_trend.get_last()
        request_rate = self.__request_rate_trend.get_last()
        ret_cost = self.__retrieval_cost_trend.get_last()
        
        return request_rate*((hit_rate*sla[1]) - ((1-hit_rate)*sla[2]) - ((1-hit_rate)*ret_cost))

class LearningThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        post_event("need_to_learn")