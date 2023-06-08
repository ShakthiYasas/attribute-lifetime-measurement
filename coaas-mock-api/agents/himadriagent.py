import os
import time
import queue
import threading
import statistics
from datetime import datetime

from agents.agent import Agent
from lib.fifoqueue import FIFOQueue_2
from lib.event import post_event_with_params
from agents.exploreagent import RandomAgent, MRUAgent, MFUAgent

import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

from agents.classifier.linkedcluster import LinkedCluster

MAX_WORKERS = 50

class HimadriAgent(threading.Thread, Agent):
    __hashLock = threading.Lock()

    def __init__(self, config, caller):
        disable_eager_execution()
        
        # Thread
        self.q = queue.Queue()
        self.timeout = 1.0/60
        super(HimadriAgent, self).__init__()

        # General Configuration
        self.__caller = caller
        self.__midtime =  (config.mid*config.window)/1000
        self.__longtime = (config.long*config.window)/1000

        # Overriding the action space as cache or not cache
        self.action_space = [0,1]

        # Desicion threshold
        self.__decision_threshold = config.e_greedy_init
        
        # Selecting the algorithem to execute during the exploration phase
        self.__explore_mentor = None
        if (config.explore_mentor.lower() == 'mru'):
            self.__explore_mentor = MRUAgent()
        elif (config.explore_mentor.lower() == 'mfu'):
            self.__explore_mentor = MFUAgent()
        else:
            self.__explore_mentor = RandomAgent()

        # History of rewards
        self.reward_history = FIFOQueue_2(100)
    
    def onThread(self, function, *args, **kwargs):
        self.q.put((function, args, kwargs))
    
    def run(self):
        while True:
            try:
                function, args, kwargs = self.q.get(timeout=self.timeout)
                _ = function(*args, **kwargs)
                # time.sleep(0.25) 
            except queue.Empty:
                self.__idle()

    def __idle(self):
        time.sleep(2.5) 

    # Select the most suitable action given a state
    # The biggest challenge here is the change of state with changes in the environment. 
    # So, there is no gurantee that the state will remain unchanged. 
    # So, the only option is to find the state that is the most similar using clustering.
    def choose_action(self, paramters): 
        observation = paramters[0]
        skip_random = paramters[1]
        ref_key = None
        if(len(paramters)>2):
            ref_key = paramters[2]

        # obs is the probability of caching.
        obs = observation['prob'] 
        entityid = observation['entityid']
        attribute = observation['attribute']

        action = 1
        if(self.__decision_threshold < obs):
            action = 0

        # print(obs)
        # print(str(action)+' for entity:'+str(entityid)+' and att:'+str(attribute))
        # print()

        if(action == 0):
            # Item is defenitly not to be cached
            delay_time = self.__calculate_delay(obs)
            post_event_with_params("subscribed_actions", (entityid, attribute, 0, delay_time, 0, observation['features'], ref_key))

            random_value = np.random.uniform()
            if(random_value <= self.__epsilons and not skip_random):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    action = self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
                    if(action != (0,0)):
                        observation = self.__caller.get_feature_vector(action[0], action[1])
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation, ref_key))
                else:
                    action = self.__explore_mentor.choose_action(self.__caller.get_observed())
                    if(action != (0,0)):
                        observation = self.__caller.get_feature_vector(action[0], action[1])
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation, ref_key)) 
        else:
            # Decided to be cached
            cached_lifetime = self.__calculate_expected_cached_lifetime(entityid, attribute, obs)
            post_event_with_params("subscribed_actions", (entityid, attribute, cached_lifetime, 0, 1, observation['features'], ref_key))

        return 0

    def __calculate_expected_cached_lifetime(self, entityid, attr, prob):
        cached_lt_res = self.__caller.get_access_to_db().read_all_with_limit('attribute-cached-lifetime',{
                    'entity': entityid,
                    'attribute': attr
                },10)
        if(cached_lt_res):
            avg_lt = statistics.mean(list(map(lambda x: x['c_lifetime'], cached_lt_res)))
            return (avg_lt/self.__decision_threshold)*(prob-self.__decision_threshold)
        else:
            return self.__midtime
    
    def __calculate_delay(self, prob):
        return (self.__longtime/self.__decision_threshold)*(self.__decision_threshold-self.__decision_threshold)
