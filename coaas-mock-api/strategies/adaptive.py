import time
import secrets
import _thread
import datetime
import threading
import statistics
import collections
from typing import Tuple

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from lib.event import post_event
from lib.event import subscribe
from agents.dqnagent import DQNAgent
from lib.fifoqueue import FIFOQueue_2
from agents.simpleagent import SimpleAgent

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

DATETIME_STRING = '%Y-%m-%d %H:%M:%S'

class Adaptive(Strategy):  
    # Dictionaries
    __delay_dict = {}
    __decision_history = {}
    __attribute_access_trend = {}
    __cached_hit_access_trend = {}
    __cached_attribute_access_trend = {}
    __temp_entity_att_provider_map = {}

    # Temporary List
    __currently_eval = set()

    # Counters
    rr_trend_size = 0
    __window_counter = 0
    __learning_counter = 0

    # Status
    __already_modified = False
    __last_item_to_recieve = None

    # Queues
    __sla_trend = FIFOQueue_2(10)
    __waiting_to_retrive = FIFOQueue_2()
    __entity_access_trend = FIFOQueue_2(100)
    __request_rate_trend = FIFOQueue_2(1000)
    __retrieval_cost_trend = FIFOQueue_2(10)
    
    __observedLock = threading.Lock()
    __decision_history_lock = threading.Lock()

    def __init__(self, db, window, isstatic=True, learncycle = 20, skip_random=False): 
        self.__db = db
        self.__cached = {}
        self.__observed = {}
        self.__evaluated = set()
        self.__last_actions = []
        self.__reqs_in_window = 0
        self.__isstatic = isstatic
        self.__local_ret_costs = []
        self.__moving_window = window
        self.__skiprandom = skip_random
        self.req_rate_extrapolation = None
        self.__learning_cycle = learncycle
        self.__current_delay_probability = 0.5
        self.__most_expensive_sla = (0.5,1.0,1.0)

        self.service_selector = ServiceSelector(db, window)
        if(not self.__isstatic):
            self.__profiler = AdaptiveProfiler(db, self.__moving_window, self.__class__.__name__.lower())

        subscribe("subscribed_actions", self.sub_cache_item)
        subscribe("subscribed_evictions", self.sub_evictor)
        subscribe("subscribed_learner", self.sub_decisioned_item)

    # Init_cache initializes the cache memory. 
    def init_cache(self):
        self.__learning_skip = max(5,self.trend_ranges[0])
        setattr(self.service_selector, 'service_registry', self.service_registry)
        setattr(self.service_selector, 'session', self.session)

        if(self.selective_cache_agent == None):
            self.selective_cache_agent = self.selective_agent_factory.get_agent()
            self.selective_cache_agent.start()
        # Set current session to profiler if not set
        if((not self.__isstatic) and self.__profiler and self.__profiler.session == None):
            self.__profiler.session = self.session
        
        setattr(self.cache_memory, 'caller_strategy', self)
        self.__is_simple_agent = isinstance(self.selective_cache_agent, SimpleAgent)

        # Extrapolation ranges 
        self.__short = self.trend_ranges[0]*(self.__moving_window/1000)
        self.__mid = self.trend_ranges[1]*(self.__moving_window/1000)
        self.__long = self.trend_ranges[2]*(self.__moving_window/1000)

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
        
        _thread.start_new_thread(self.service_registry.update_ret_latency, (self.service_selector.get_current_retrival_latency(),))

        if(self.__window_counter >= self.trend_ranges[1]): 
            exp_time = datetime.datetime.now()-datetime.timedelta(seconds=self.__moving_window/1000)
            self.__clear_observed(exp_time)
            self.__clear_cached()
            if(not self.__already_modified and self.__is_simple_agent):
                _thread.start_new_thread(self.selective_cache_agent.modify_dicount_rate, (False,))
                _thread.start_new_thread(self.selective_cache_agent.modify_epsilon, (self.__learning_counter,))
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
            if(len(value['req_ts'])>0 and value['req_ts'][-1] <= exp_time):
                # The entire entity hasn't been accessed recently
                if(key in self.__observed):
                    del self.__observed[key]
                if(key in self.__attribute_access_trend):
                    del self.__attribute_access_trend[key]
            else:
                self.__update_attribute_access_trend(exp_time, key, value, __reqs_in_window)

    def __update_attribute_access_trend(self, exp_time, key, value, __reqs_in_window=None):
        __reqs_in_window = self.__reqs_in_window if not __reqs_in_window else __reqs_in_window
        invalidtss = [num for num in value['req_ts'] if num < exp_time]
        for i in range(0,len(invalidtss)):
            value['req_ts'].pop(0)

        validtss = value['req_ts']
        access_freq = 0
        if(__reqs_in_window>0):
            access_freq = 1 if len(validtss) > __reqs_in_window else len(validtss)/__reqs_in_window 
        self.__entity_access_trend.push(access_freq)
                
        for curr_attr, access_list in value['attributes'].items():
            invalidtss = [num for num in access_list if num <= exp_time]
            for i in range(0,len(invalidtss)):
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
        prev_decisions = self.__decision_history.copy().items()   
        for action, values in prev_decisions:
            entityid = action[0]
            att = action[1]
            if((not values[2]) and ((values[1] == 0 and values[3] <= self.__window_counter + self.__learning_skip) 
                    or (values[1]>0 and entityid in self.__cached_hit_access_trend and 
                        att in self.__cached_hit_access_trend[entityid] and 
                        self.__cached_hit_access_trend[entityid][att].isfull()))):
                self.__decision_history_lock.acquire()
                updated_val = list(values)
                updated_val[2] = True
                self.__decision_history[action] = tuple(updated_val)
                self.__decision_history_lock.release()

                curr_state = self.__translate_to_state(action[0], action[1])
                diff = self.__window_counter - values[3]
                reward = self.__calculate_reward(action[0], action[1], values[0], diff)  
                if(not self.__is_simple_agent):
                    self.selective_cache_agent.onThread(self.selective_cache_agent.learn, (values[0], values[1], reward[0], curr_state, values[4]))
                else:
                    self.selective_cache_agent.set_to_reward_history(reward[0])

                self.__decision_history_lock.acquire()
                del self.__decision_history[action]
                self.__decision_history_lock.release()
    
    def __update_reward_evicted_early(self, entity, attribute):
        action = (entity, attribute)
        if(action in self.__decision_history):
            decision = self.__decision_history[action]
            diff = self.__window_counter - decision[3]

            curr_state = self.__translate_to_state(action[0], action[1])

            if(curr_state == None):
                reward = -10
                if(not self.__is_simple_agent):
                    self.selective_cache_agent.onThread(self.selective_cache_agent.learn, (decision[0], decision[1], reward, {'features':decision[0]}, decision[4]))
                else:
                    self.selective_cache_agent.set_to_reward_history(reward)
            else:
                if(self.__window_counter >= decision[3]+3):
                    reward = self.__calculate_reward(action[0], action[1], decision[0], diff)  

                    if(not self.__is_simple_agent):
                        self.selective_cache_agent.onThread(self.selective_cache_agent.learn, (decision[0], decision[1], reward[0], curr_state, decision[4]))
                    else:
                        self.selective_cache_agent.set_to_reward_history(reward[0])
                else:
                    past_sla = self.__sla_trend.get_last_range(diff)
                    past_ret_costs = self.__retrieval_cost_trend.get_last_range(diff)
                    past_request_rate = self.__request_rate_trend.get_last_range(diff)
                    
                    total_gain = 0
                    total_requests = 0

                    if(entity in self.__cached_attribute_access_trend and 
                            attribute in self.__cached_attribute_access_trend[entity]):
                        # This item has been cached in the last decision epoch 
                        past_hr = self.__cached_hit_access_trend[entity][attribute].get_last_range(diff)
                        past_ar = self.__cached_attribute_access_trend[entity][attribute].get_last_range(diff)

                        for idx in range(0,len(past_ar)):
                            price = past_hr[idx]*past_sla[idx][1]
                            penalty = (1-past_hr[idx])*past_sla[idx][2]*self.__current_delay_probability
                            out = (1/past_request_rate[idx])*past_ar[idx]
                            retrieval = (1-past_hr[idx])*past_ret_costs[idx]

                            total_requests += out
                            total_gain += out*(price - penalty - retrieval)
                        
                        # This returns the gain or loss of caching an item per request
                        reward = round(total_gain/total_requests,3) if total_requests>0 else -10
                        if(not self.__is_simple_agent):
                            self.selective_cache_agent.onThread(self.selective_cache_agent.learn, (decision[0], decision[1], reward[0], curr_state, decision[4]))
                        else:
                            self.selective_cache_agent.set_to_reward_history(reward[0])

            if(action in self.__decision_history):
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
                    most_exp = list(self.__most_expensive_sla)
                    if(fthresh[2]>self.__most_expensive_sla[2]):
                        most_exp[2] = fthresh[2]
                    if(fthresh[1]<self.__most_expensive_sla[1]):
                        most_exp[1] = fthresh[1]
                    self.__most_expensive_sla = tuple(most_exp)

        now = datetime.datetime.now()

        output = {}
        for ent in json:
            refetching = [] # Freshness not met for the value generated by a producer [(entityid, prodid)]
            new_context = [] # Need to fetch the entity with all attributes [(entityid, [attributes])]
            new_providers = set() # The speciifc provider's which may be required to respond to the query which is not available in cache
            attrs_need_providers = []

            # Check freshness of requested attributes
            entityid = ent['entityId']
            conditions = []
            if('conditions' in ent):
                conditions = ent['conditions']
            lifetimes = None
            if(self.__isstatic):
                lifetimes = self.service_registry.get_context_producers(entityid,ent['attributes'],conditions)          

            cache_result = self.cache_memory.run('get_statistics_all')
            if(entityid in cache_result):
                reference = None
                # Entity is cached
                # Atleast one of the attributes of the entity is already cached 
                if(not (entityid in self.__cached)):
                    self.__cached[entityid] = {}

                output[entityid] = {}
                # Refetch from the producer if atleast 1 of it's attributes are not available or not fresh
                is_all_cached, uncached = self.cache_memory.run('are_all_atts_cached', (entityid, ent['attributes']))

                if(is_all_cached):
                    self.__learning_counter -= 1
                    # All of the attributes requested are in cache for the entity
                    for att_name in ent['attributes']:
                        # Get all values from the context producers for the attribute in cache
                        att_in_cache = self.cache_memory.run('get_value_by_key', (entityid, att_name))
                        ishit = 1
                        if(not self.__isstatic):
                            for prodid,val,lastret,rec_bit in att_in_cache:
                                # Estimated lifetime of the attribute
                                lastret = datetime.datetime.fromtimestamp(time.mktime(time.strptime(lastret, DATETIME_STRING)))
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
                            #print('Entity = '+ str(entityid) + ' and attribute = '+ att_name)
                            for prodid,val,lastret,rec_bit in att_in_cache:
                                lastret = datetime.datetime.fromtimestamp(time.mktime(time.strptime(lastret, DATETIME_STRING)))
                                if(prodid in lifetimes):
                                    lt = lifetimes[prodid]['lifetimes'][att_name]
                                    #print('Lifetime: '+str(lt))
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
                            
                            # Checking if there exist new providers that are already not cached
                            l1 = lifetimes.keys()
                            l2 = [info[0] for info in att_in_cache]
                            # print()
                            new_pros = set(l1) - set(l2)
                            if(len(new_pros)):
                                # There are some new context providers from which data need to be retrieved
                                new_providers.add((entityid,att_name))
                                attrs_need_providers += list(new_pros)
                                            
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
                        # Update hit rate here for those which have already been cached of the entity
                        att_in_cache = set(ent['attributes']) - uncached
                        
                        for att_name in att_in_cache:
                            ishit = 1
                            pro_att = self.cache_memory.run('get_value_by_key', (entityid, att_name))
                            for prodid,val,lastret,rec_bit in pro_att:
                                if(prodid in lifetimes):
                                    lt = lifetimes[prodid]['lifetimes'][att_name]
                                    if(lt >= 0):
                                        extime = lt * (1 - fthresh[0])
                                        lastret = datetime.datetime.fromtimestamp(time.mktime(time.strptime(lastret, DATETIME_STRING)))
                                        time_at_expire = lastret + datetime.timedelta(seconds=extime)
                                        if(now > time_at_expire):
                                            ishit = 0

                            if(not (att_name in self.__cached[entityid])):
                                self.__cached[entityid][att_name] = [ishit]
                            else:
                                self.__cached[entityid][att_name].append(ishit)

                            if(self.__is_simple_agent):
                                new_pros = set(lifetimes.keys()) - set([info for info in att_in_cache])
                                if(len(new_providers)):
                                    # There are some new context providers from which data need to be retrieved
                                    new_providers.add((entityid,att_name))
                                    attrs_need_providers += list(new_pros)

                        reference = secrets.token_hex(nbytes=8)
                        self.__temp_entity_att_provider_map[reference] = list(lifetimes.keys())
                        caching_attrs = self.__evalute_attributes_for_caching(entityid, uncached, reference)
                        if(caching_attrs):
                            new_context.append((entityid,caching_attrs,lifetimes))
                            uncached = set(uncached) - set(caching_attrs)

                        self.__evaluated.add(entityid)
                    else:
                        self.__update_observed(entityid, ent['attributes'])
                    
                threads = []
                if(new_providers):
                    en_re_th = threading.Thread(target=self.__cache_entity_attribute_pairs(list(new_providers), providers=attrs_need_providers))
                    en_re_th.start()
                    threads.append(en_re_th)
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

                val = self.cache_memory.run('get_values_for_entity', (entityid, ent['attributes']))
                
                for att_name,prod_values in val.items():
                    if(prod_values != None):
                        output[entityid][att_name] = [res[1] for res in prod_values]
                    else:
                        # This is when the entity is not even evaluated to be cached
                        if(lifetimes):
                            url_list = list(map(lambda k: (k[0],k[1]['url']), lifetimes.items()))
                            res = self.service_selector.get_response_for_entity([att_name], url_list)  
                            output[entityid][att_name] = [x[1] for x in res[att_name]]
                            # Update observed
                            self.__observedLock.acquire()
                            self.__update_observed(entityid, [att_name])
                            self.__observedLock.release()
            else:
                # Even the entity is not cached previously
                # So, first retrieving the entity
                if(lifetimes):
                    reference = secrets.token_hex(nbytes=8)
                    self.__temp_entity_att_provider_map[reference] = list(lifetimes.keys())
                    out = self.__retrieve_entity(ent['attributes'],lifetimes)
                    output[entityid] = {}
                    for att_name,prod_values in out.items():
                        output[entityid][att_name] = [res[1] for res in prod_values]
                        
                    self.__learning_counter += 1
                    # Evaluate whether to cache               
                    # Doesn't cache any item until atleast the mid range is reached
                    #_thread.start_new_thread(self.__evaluate_and_updated_observed_in_thread, (entityid, out))
                    self.__evaluate_and_updated_observed_in_thread(entityid, out, reference)

        if(self.__learning_counter % self.__learning_cycle == 0):
            if(isinstance(self.selective_cache_agent, DQNAgent)):
                # Trigger learning for the DQN Agent
                self.trigger_agent_learning()
            elif(isinstance(self.selective_cache_agent, SimpleAgent)):
                # Modify the discount rate
                self.__already_modified = True
                _thread.start_new_thread(self.selective_cache_agent.modify_dicount_rate, ())
                _thread.start_new_thread(self.selective_cache_agent.modify_epsilon, (self.__learning_counter,))

        return output

    def __evaluate_and_updated_observed_in_thread(self, entityid, out, ref_key):
        self.__observedLock.acquire()
        self.__update_observed(entityid, list(out.keys()))
        self.__observedLock.release()

        if(self.__window_counter >= self.trend_ranges[1]+1):
            # Run this in the background
            self.__evaluate_for_caching(entityid, out, ref_key)
            self.__evaluated.add(entityid)
            

    ###################################################################################
    # Section 03 - Evluating to Cache
    # This section performs the actions which are required to estimate the value of 
    # caching a context item. 
    ###################################################################################

    # Evaluate for caching
    def __evalute_attributes_for_caching(self, entityid, attributes:list, ref_key) -> list:
        # Evaluate the attributes to cache or not
        actions = []
        now = datetime.datetime.now()
        for att in attributes:
            if((self.__check_delay(entityid, att) or self.__is_spike(entityid, att)) and not self.__waiting_to_retrive.is_enqued(entityid, att)):
                # print('There is a spike or no delay restriction!')
                # Translate the entity,attribute pair to a state
                observation = self.__translate_to_state(entityid,att)
                if(observation==None):
                    if(entityid in self.__delay_dict):
                        self.__delay_dict[entityid][att] = self.__window_counter + 1
                    else:
                        self.__delay_dict[entityid] = { att: self.__window_counter + 1 }
                else:
                    # Select the action for the state using the RL Agent
                    action, (est_c_lifetime, est_delay) = None, (None, None)
                    if(self.__is_simple_agent):
                        action, (est_c_lifetime, est_delay) = self.selective_cache_agent.choose_action(observation, skip_random=self.__skiprandom)                
                        if(action != (0,0)):
                            wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                            self.cache_memory.run('addcachedlifetime', (action, wait_time))
                            self.__last_actions.append(action)
                            actions.append(action[1])
                        else:
                            if(entityid in self.__delay_dict):
                                self.__delay_dict[entityid][att] = self.__window_counter + est_delay
                            else:
                                self.__delay_dict[entityid] = { att: self.__window_counter + est_delay }
                        
                        if(action != (0,0) and action[0] == entityid):
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.run('addcachedlifetime', (action, wait_time))
                                self.__last_actions.append(action)
                                actions.append(action[1])
                        else:
                            if(action != (0,0)):
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.run('addcachedlifetime', (action, wait_time))
                                self.__last_actions.append(action)
                                self.__cache_entity_attribute_pairs([action])
                            else:
                                if(entityid in self.__delay_dict):
                                    self.__delay_dict[entityid][att] = self.__window_counter + est_delay
                                else:
                                    self.__delay_dict[entityid] = { att: self.__window_counter + est_delay }
                    else:
                        # if(not((entityid, att) in self.__currently_eval)):
                        self.__currently_eval.add((entityid,att))
                        self.selective_cache_agent.onThread(self.selective_cache_agent.choose_action, (observation, self.__skiprandom, ref_key))                

        return actions         

    def __evaluate_for_caching(self, entityid, attributes:dict, ref_key):
        # Check if this entity has been evaluated for caching in this window
        # Evaluate if the entity can be cached
        random_one = []
        is_caching = False
        updated_attr_dict = []
        now = datetime.datetime.now()
        if(not (entityid in self.__evaluated)):
            # Entity hasn't been evaluted in this window before
            for att in attributes.keys():
                checker = (self.__check_delay(entityid, att) or self.__is_spike(entityid, att)) and not self.__waiting_to_retrive.is_enqued(entityid, att)
                # print('There is a spike or no delay restriction!')
                if(checker):
                    observation = self.__translate_to_state(entityid,att)
                    if(observation == None):
                        if(entityid in self.__delay_dict):
                            self.__delay_dict[entityid][att] = self.__window_counter + 1
                        else:
                            self.__delay_dict[entityid] = { att: self.__window_counter + 1 }
                    else:
                        action, (est_c_lifetime, est_delay) = None, (None, None)
                        if(self.__is_simple_agent):
                            action, (est_c_lifetime, est_delay) = self.selective_cache_agent.choose_action(observation, skip_random=self.__skiprandom)

                            if(action != (0,0) and action[0] == entityid):
                                updated_attr_dict.append(action[1])
                                wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                                self.cache_memory.run('addcachedlifetime', (action, wait_time))
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
                            #if(not((entityid, att) in self.__currently_eval)):
                            self.__currently_eval.add((entityid,att))
                            self.selective_cache_agent.onThread(self.selective_cache_agent.choose_action, (observation, self.__skiprandom, ref_key))  
                                          
        if(is_caching):
            # Add to cache 
            self.__last_actions += [(entityid, x) for x in updated_attr_dict]
            updated_dict = {}
            for att in updated_attr_dict:
                updated_dict[att] = attributes[att]
            # print('From Evlauting for cache!')
            # print('Saving from _evalute for cache!')
            self.cache_memory.run('save', (entityid,updated_dict))
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:updated_attr_dict})  
            
            for att in updated_attr_dict:
                if(entityid in self.__attribute_access_trend and att in self.__attribute_access_trend[entityid]):
                    del self.__attribute_access_trend[entityid][att]
                if(entityid in self.__observed and att in self.__observed[entityid]):
                    del self.__observed[entityid][att]
            if(entityid in self.__observed and self.__observed[entityid]):
                del self.__observed[entityid]

        if(random_one):
            for entity, att, cachelife in random_one:
                
                wait_time = now + datetime.timedelta(seconds=cachelife)
                self.cache_memory.run('addcachedlifetime', (action, wait_time))
                self.__last_actions.append((entity, att))
            self.__cache_entity_attribute_pairs(random_one, is_random=True)
    
    # Check if the context attribute has been evaluated to be delayed for caching
    def __check_delay(self,entity,attr):
        if(entity in self.__delay_dict and attr in self.__delay_dict[entity]):
            if(self.__delay_dict[entity][attr] == -1): return False
            if(self.__window_counter <= self.__delay_dict[entity][attr]): return False
            else: 
                del self.__delay_dict[entity][attr]
                if(not self.__delay_dict[entity]):
                    del self.__delay_dict[entity]
                # print('Entity has elapsed delay time!')
                return True
        else: 
            # print('Entity is not delayed!')
            return True
    
    # Check if the context attribute is observed to show a spike in demand
    def __is_spike(self,entity,attr):
        # print('Entity: '+ str(entity) + ' attribute: '+ str(attr))
        # print('Is observed: '+ str(self.__isobserved(entity, attr)))
        # print('Entity in Access trend: '+ str(entity in self.__attribute_access_trend))
        # if(entity in self.__attribute_access_trend):
        #     print('Attribute in Access Trend: '+ str(attr in self.__attribute_access_trend[entity]))

        if(self.__isobserved(entity, attr) and entity in self.__attribute_access_trend
                and attr in self.__attribute_access_trend[entity]):
            att_trend = self.__attribute_access_trend[entity][attr].get_last_range(2)
            # print(att_trend)
            if(len(att_trend)<2):
                return False
            if(att_trend[1] > 0):
                growth = (att_trend[1] - att_trend[0])/att_trend[1]
                # print('Growth: '+ str(growth))
                if(growth >= 10):
                    return True
            return False
        else: 
            # print('Entity: '+ str(entity) + ' attribute: '+ str(attr) + ' is not in the observed list and/or the access trend!')
            return False  

    def sub_decisioned_item(self, parameters):
        entity = parameters[0]
        attribute = parameters[1]
        observation = parameters[5]
        action = parameters[4]

        self.__decision_history_lock.acquire()
        self.__decision_history[(entity, attribute)] = (observation, action, False, self.__window_counter, parameters[2] if action == 1 else parameters[3])
        self.__decision_history_lock.release()

    # Subcriber method to cache an item
    def sub_cache_item(self, parameters):
        entity = parameters[0]
        ref_key = parameters[6] 

        if(self.__last_item_to_recieve == None):
            self.__waiting_to_retrive.push(parameters)
        elif(entity != self.__last_item_to_recieve[0] or ref_key != self.__last_item_to_recieve[6]): 
            # Different entity and reference 
            self.__waiting_to_retrive.push(parameters)
            last_observed = self.__last_item_to_recieve
            providers = []
            if(last_observed[6] in self.__temp_entity_att_provider_map):
                providers =  self.__temp_entity_att_provider_map[last_observed[6]]

            now = datetime.datetime.now()
            ent_att_pairs = []
            
            all_ent_atts = list(filter(lambda x: x[0] == last_observed[0] and x[6] == last_observed[6], self.__waiting_to_retrive.getlist()))

            for current_element in all_ent_atts:
                entity = current_element[0]
                attribute = current_element[1]
                est_c_lifetime = current_element[2]
                est_delay = current_element[3]
                action = current_element[4]
                observation = current_element[5]

                if(action != 0):
                    # The item is to be cached
                    # Could be a random one or an item evulated to be cached
                    wait_time = now + datetime.timedelta(seconds=est_c_lifetime)
                    self.cache_memory.run('addcachedlifetime', ((entity, attribute), wait_time))
                    self.__last_actions.append((entity, attribute))
                    ent_att_pairs.append((entity, attribute))
                else:
                    self.__observedLock.acquire()
                    if(entity in self.__delay_dict):
                        self.__delay_dict[entity][attribute] = self.__window_counter + est_delay
                    else:
                        self.__delay_dict[entity] = { attribute: self.__window_counter + est_delay }
                        self.__update_observed(entity, attribute)
                        self.__observedLock.release()     

                self.__decision_history[(entity, attribute)] = (observation, action, False, self.__window_counter, est_c_lifetime if action > 0 else est_delay)
                if (entity,attribute) in self.__currently_eval: 
                    self.__currently_eval.remove((entity,attribute))
            
            self.__waiting_to_retrive.remove_items(all_ent_atts)

            if(ent_att_pairs):
                self.__cache_entity_attribute_pairs(ent_att_pairs, providers=list(providers)) 
            if(last_observed[6] in self.__temp_entity_att_provider_map[last_observed[6]]):
                del self.__temp_entity_att_provider_map[last_observed[6]]
        else:
            # Same entity
            self.__waiting_to_retrive.push(parameters)

        self.__last_item_to_recieve = parameters

    # Subcriber method to evict an item
    def sub_evictor(self, parameters):
        entity = parameters[0]
        attribute = parameters[1] 

        if(attribute == None):
            if(entity in self.__cached):
                del self.__cached[entity]
            if(entity in self.__cached_hit_access_trend):
                del self.__cached_hit_access_trend[entity]
            if(entity in self.__cached_attribute_access_trend):
                del self.__cached_attribute_access_trend[entity]
        else:
            if(entity in self.__cached and attribute in self.__cached[entity]):
                del self.__cached[entity][attribute]
            if(entity in self.__cached_hit_access_trend and attribute in self.__cached_hit_access_trend[entity]):
                del self.__cached_hit_access_trend[entity][attribute]
            if(entity in self.__cached_attribute_access_trend and attribute in self.__cached_attribute_access_trend[entity]):
                del self.__cached_attribute_access_trend[entity][attribute]
            self.__update_reward_evicted_early(entity, attribute)
            

    ###################################################################################
    # Section 04 - Caching
    # This section performs the caching actions by calling the methods in cache memory
    # instance and updating the statistics. 
    ###################################################################################
       
    def __cache_entity_attribute_pairs(self, entityttpairs, is_random=False, providers = []):
        ent_att = {}
        lifetimes = {}
        for pair in entityttpairs:
            if(pair[0] in ent_att):
                if(pair[1] in ent_att[pair[0]]):
                    ent_att[pair[0]].append(pair[1])
                else:
                    ent_att[pair[0]] = [pair[1]]
            else:
                ent_att[pair[0]] = [pair[1]]

        for entityid, attlist in ent_att.items():
            if(providers):
                lifetimes = self.service_registry.get_context_producers_by_ids(providers)
            else:
                lifetimes = self.service_registry.get_context_producers(entityid,attlist)
            li = [(prodid,value['url']) for prodid, value in lifetimes.items()]
            response = self.service_selector.get_response_for_entity(attlist,li)
            # print('From Caching entity attribute pair')
            # print('Saving from __cache_entity_attribute_pairs!')
            self.cache_memory.run('save', (entityid, response))
            
            for att in attlist:
                if(entityid in self.__cached):
                    self.__cached[entityid][att] = [False]
                else:
                    self.__cached[entityid] = {att:[False]}
                # Push to profiler
                if(not self.__isstatic):
                    self.__profiler.reactive_push({entityid:[att]})    
                if(entityid in self.__attribute_access_trend and att in self.__attribute_access_trend[entityid]):
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

    def get_feature_vector(self, entityid, att):
        return self.__translate_to_state(entityid, att)

    # Select the closet range boundary for the current state 
    def __closest_point(self, c_life):
        s_dis = abs(self.__short - c_life)
        m_dis = abs(self.__mid - c_life)
        l_dis = abs(self.__long - c_life)

        closest = self.__short
        if(m_dis<s_dis or m_dis==s_dis):
            closest = self.__mid
            if(l_dis<m_dis or l_dis==m_dis):
                closest = self.__long

        return closest

    def __expected_life(self, lt_list):
        exp_life_dict = {}
        for c_life in lt_list:
            closest = self.__closest_point(c_life)
            if(closest in exp_life_dict):
                exp_life_dict[closest] += 1
            else: 
                exp_life_dict[closest] = 1
        
        list_len = len(lt_list)
        exp_life_dict.update((k,v/list_len) for k,v in exp_life_dict)
        
        out_1 = collections.OrderedDict(sorted(exp_life_dict.items()))
        expected_life = sorted(out_1.items(), key=lambda item: item[1])[-1][0]

        return expected_life
        
    # Translating an observation to a state
    def __translate_to_state(self, entityid, att):
        isobserved = self.__isobserved(entityid, att)
        iscached = self.cache_memory.run('get_statistics',(entityid, att))
        # Access Rates [0-5]
        fea_vec = self.__calculate_access_rates(isobserved, entityid, att)
        # Hit Rates and Expectations [6-11]
        lifetimes = self.service_registry.get_context_producers(entityid,[att])
        # The above step could be optimzed by using a view (in SQL that updates by a trigger)
        new_feas, avg_latency = self.__calculate_hitrate_features(bool(not iscached and isobserved), entityid, att, lifetimes)
        
        if((new_feas, avg_latency) == (None, None)):
            return None

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
                fea_vec.append(round(avg_lts,3))
            else:
                fea_vec.append(round(statistics.mean(avg_lts),3))
        else:
            fea_vec.append(0)

        # Latency [13]
        fea_vec.append(round(avg_latency,3))

        # Average Retriveal Cost [14]
        avg_ret_cost = statistics.mean([values['cost'] for values in lifetimes.values()]) if(lifetimes) else 9999
        fea_vec.append(round(avg_ret_cost,3))

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
            if(not (entityid in self.__attribute_access_trend) or not (att in self.__attribute_access_trend[entityid])):
                exp_time = datetime.datetime.now()-datetime.timedelta(seconds=self.__moving_window/1000)
                self.__update_attribute_access_trend(exp_time, entityid, self.__observed[entityid])

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
                fea_vec.append(round(attribute_trend.get_last_position(ran),3))
                exp_ar = extrapolation[trend_size-2+ran]
                if(exp_ar < 0):
                    exp_ar = 0
                elif(exp_ar > 1):
                    exp_ar = 1
                fea_vec.append(round(exp_ar,3))

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
                fea_vec.append(round(attribute_trend.get_last_position(ran),3))
                exp_ar = extrapolation[trend_size-2+ran]
                if(exp_ar < 0):
                    exp_ar = 0
                elif(exp_ar > 1):
                    exp_ar = 1
                fea_vec.append(round(exp_ar,0))
                
        else:
            # Actual and Expected Access Rates 
            # No actual or expected access rates becasue the item is already cached
            fea_vec = [0,0,0,0,0,0]

        return fea_vec
    
    # Calculates the hit rates for the given observation
    def __calculate_hitrate_features(self, isnotcached, entityid, att, lifetimes):
        fea_vec = []
        avg_rt = self.service_selector.get_average_responsetime_for_attribute(list(map(lambda x: x[0], lifetimes.items()))) if lifetimes else 9999

        if(isnotcached):
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
                    fea_vec.append(round(hit_trend.get_last_position(ran),3))
                    exp_hr = extrapolation[trend_size-2+ran]
                    if(exp_hr < 0):
                        exp_hr = 0
                    elif(exp_hr > 1):
                        exp_hr = 1
                    fea_vec.append(round(exp_hr,3))
            else:
                # The item is not in cache or not observed either.
                # This means the item is very new. So, delay it by 1 window.
                # Expected Hit Rate (If Not Cached)
                return None, None
                
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
            #print('From refreshing entity')
            # print('Saving from __refresh_cache_for_entity!')
            self.cache_memory.run('save', (entityid,response))
            # Push to profiler
            if(not self.__isstatic):
                self.__profiler.reactive_push({entityid:response})

    # Refreshing for selected context producers
    def __refresh_cache_for_producers(self, refresh_context) -> None:
        # Retrive raw context from provider according to the provider
        for entityid,attribute_list,prodid,url in refresh_context:
            response = self.service_selector.get_response_for_entity(attribute_list,[(prodid,url)])
            # Save items in cache
            # print('From Refreshing context provider')
            # print('Saving from __refresh_cache_for_producers!')
            self.cache_memory.run('save', (entityid,response))
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
                    time_between_2_requests = 1/req_at_point
                    if(delta >= (time_between_2_requests*2)):
                        # It is effective to cache
                        # because multiple requests can be served
                        exp_hr = (delta*req_at_point)/((delta*req_at_point)+1)
                        fea_vec.append(round(exp_hr,3))
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
        return list(set(attributes) - set(self.cache_memory.run('get_attributes_of_entity', (entityid,))))

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
                penalty = (1-past_hr[idx])*past_sla[idx][2]*self.__current_delay_probability
                out = (1/past_request_rate[idx])*past_ar[idx] if past_request_rate[idx] > 0 else 0
                retrieval = (1-past_hr[idx])*past_ret_costs[idx]

                total_requests += out
                total_gain += out*(price - penalty - retrieval)
            
            # This returns the gain or loss of caching an item per request
            reward = round(total_gain/total_requests,3) if total_requests>0 else -10

            return reward, is_cached
        else:
            # This item was not cached
            expected_vals = []

            expected_ar = []
            short_inc = (previous_state[1] - previous_state[0])/self.trend_ranges[0] if self.trend_ranges[0] > 0 else 0
            mid_inc = (previous_state[3] - previous_state[1])/(self.trend_ranges[1] - self.trend_ranges[0]) if (self.trend_ranges[1] - self.trend_ranges[0])  > 0 else 0
            long_inc = (previous_state[5] - previous_state[3])/(self.trend_ranges[2] - self.trend_ranges[1]) if (self.trend_ranges[2] - self.trend_ranges[1]) > 0 else 0
            
            curr_ar = previous_state[1]
            for i in range(0,self.trend_ranges[2]):
                if(i < self.trend_ranges[0]):
                    curr_ar += short_inc
                elif(self.trend_ranges[0] <= i < self.trend_ranges[1]):
                    curr_ar += mid_inc
                else:
                    curr_ar += long_inc
                expected_ar.append(curr_ar)

            expected_hr = []
            short_inc = (previous_state[6] - previous_state[7])/self.trend_ranges[0] if self.trend_ranges[0] > 0 else 0
            mid_inc = (previous_state[9] - previous_state[7])/(self.trend_ranges[1] - self.trend_ranges[0]) if (self.trend_ranges[1] - self.trend_ranges[0]) > 0 else 0
            long_inc = (previous_state[11] - previous_state[9])/(self.trend_ranges[2] - self.trend_ranges[1]) if (self.trend_ranges[2] - self.trend_ranges[1]) > 0 else 0

            curr_hr = previous_state[1]
            for i in range(0,self.trend_ranges[2]):
                if(i < self.trend_ranges[0]):
                    curr_hr += short_inc
                elif(self.trend_ranges[0] <= i < self.trend_ranges[1]):
                    curr_hr += mid_inc
                else:
                    curr_hr += long_inc
                expected_hr.append(curr_hr)
            
            # Expected Values
            for idx in range(1,diff):
                price = expected_hr[idx]*past_sla[idx][1]  
                penalty = (1-expected_hr[idx])*past_sla[idx][2]*self.__current_delay_probability
                retrieval = (1-expected_hr[idx])*past_ret_costs[idx]
                out = (1/past_request_rate[idx])*expected_ar[idx] if past_request_rate[idx] > 0 else 0

                total_requests += out
                expected_vals.append(out*(price - penalty - retrieval))

            # Observed Values
            observed_vals = []
            past_ar = self.__attribute_access_trend[entityid][att].get_last_range(diff)
            for idx in range(0,len(past_ar)):            
                penalty = past_sla[idx][2]*self.__current_delay_probability
                retrieval = past_ret_costs[idx]
                out = (1/past_request_rate[idx])*past_ar[idx] if past_request_rate[idx] > 0 else 0

                total_requests += out
                observed_vals.append(out*(0 - penalty - retrieval))

            diff = min(len(observed_vals), len(expected_vals), diff)
            diff = sum([observed_vals[i] - expected_vals[i] for i in range(0,diff)])
            reward = round(diff/total_requests,3) if total_requests>0 else 10
            
            # This returns the regret of not caching the item
            return reward, is_cached

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
        res = self.cache_memory.run('get_statistics_entity', (entityid,))
        output = {
                'cached_attributes': [i for i in res.keys()] if res else {},
                'epsilon' : self.selective_cache_agent.get_current_epsilon(),
                'discount_rate' : self.selective_cache_agent.get_discount_rate()
            }

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
                hit_rate = self.cache_memory.run('get_last_hitrate', (10,))

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
            hr = self.cache_memory.run('get_last_hitrate',(1,))[0]
            hit_rate = hr[0] if isinstance(hr,Tuple) else hr
        
        sla = self.__sla_trend.get_last()
        request_rate = self.__request_rate_trend.get_last()
        ret_cost = self.__retrieval_cost_trend.get_last()
        
        return request_rate*((hit_rate*sla[1]) - ((1-hit_rate)*sla[2]) - ((1-hit_rate)*ret_cost))

    def get_access_to_db(self):
        return self.__db

    def get_currently_cached_entities(self):
        return list(self.cache_memory.run('get_statistics_all').keys())

    def get_hit_rate_variation(self):
        return self.cache_memory.run('get_hitrate_trend')

    def is_waiting_to_retrive(self, entity, attribute):
        self.__waiting_to_retrive

class LearningThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        post_event("need_to_learn")