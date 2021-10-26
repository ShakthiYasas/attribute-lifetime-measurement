import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import time
import datetime
import threading

from profiler import Profiler
from lib.event import post_event

# Profiler class for adaptive lifetime based refreshing
# Perform the inferencing of average lifetime by intercepting the responses from the context providers.
class AdaptiveProfiler(Profiler):
    # Class variables
    __db = None

    __lookup = {} # Index look up for each context attribute
    __mean = [] # Contains the current average inferred lifetime of each attribute
    __freshness_reqiurement = {} # Current level of freshness requirements
    
    # Static configurations for background threads 
    __interval = 1.0 # 1 Second
    __threadpool = []

    last_time = datetime.datetime.now()

    def __init__(self, attributes, db, window, caller_name, session = None):
        # Instance variables
        index = 0
        self.__db = db   
        self.window = window
        self.session = session
        self.caller_name = caller_name # Retrieval strategy
        self.__mean = [0] * len(attributes) # Mean lifetime of all attributes initialized to 0
        self.__most_recently_used = [[]] * len(attributes) # Zero matrix initialized
        
        # Entring values to lookup table
        for att in attributes:
            self.__lookup[att] = index
            index+=1
        
        # Initializing background thready to clear collected responses that fall outside the window.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 

    def run(self):
        while True:
            self.clear_expired()
            time.sleep(self.__interval)

    # Get the freshness of the most SLA for each attribute
    def update_freshness_requirement(self,freshnesses):
        for key, value in freshnesses.items():
            self.__freshness_reqiurement[self.__lookup[key]] = value

    # Function to refresh greedily based on event-trigger
    def auto_cache_refresh_for_greedy(self, attributes) -> None:
        for att in attributes:
            th = GreedyRetrievalThread(self.__lookup[att], att, self)
            self.__threadpool.append(th)
            th.start()
    
    # Clear function that run on the background
    def clear_expired(self) -> None:
        exp_time = datetime.datetime.now() - datetime.timedelta(milliseconds=self.window)
        for row in self.__most_recently_used:
            for stamp in row:
                # If value of older than the oldest in the current window, clear it from the matrix
                if(len(stamp) != 0 and stamp[1] < exp_time):
                    row.remove(stamp)

    # Reactive push recomputes the moving avergae lifetime of the 
    # responses recived and refreshes the cache entry.
    def reactive_push(self, response, is_greedy=False) -> None:
        curr_time = datetime.datetime.now()
        
        # Updating the statistics of all the attributes that has been retrived
        for key,value in response.items():
            if(key == 'step'):
                continue

            idx = self.__lookup[key]
            lst_vals = self.__most_recently_used[idx]
            
            # Calculating the time gap with last retrieval
            duration = 0 
            if not lst_vals:
                duration = ((curr_time - self.last_time).total_seconds())*1000.0
            else:
                duration = ((curr_time - lst_vals[-1][1]).total_seconds())*1000.0
            
            # Update the most recent data matrix with the recent retrieval 
            self.__most_recently_used[idx].append((value, curr_time, duration))
            
            # Calculate the new moving average lifetime
            mean = self.calculate_meanlife(idx, is_greedy)
            
            self.__mean[idx] = mean
            self.__db.insert_one(key+'-lifetime',{'session': self.session, 'strategy': self.caller_name, 'lifetime:':mean, 'time': curr_time, 'step': response['step']})    
        
        self.last_time = curr_time

    # Calculates the moving average lifetime of a context attribute
    def calculate_meanlife(self, idx, is_greedy) -> float:
        count = 0
        total_sum = 0
        local_sum = 0
        curr_val = None

        for item in self.__most_recently_used[idx]:
            if(curr_val == None):
                curr_val = item[0]
                local_sum = item[2]
            else:
                if(item[0] != curr_val):
                    count += 1
                    total_sum += local_sum
                    curr_val = item[0]
                    local_sum = item[2]
                else:
                    local_sum += item[2]
        
        # Adjusting for initial state
        total_sum += local_sum
        count += 1

        # Adjusting greedy algorithm to manage with the most expensive SLA
        # However, considering the lifetime until freshness = 0 if no freshness requirement is yet specificed.
        if(is_greedy and idx in self.__freshness_reqiurement):
            return (total_sum/count) * (1 - self.__freshness_reqiurement[idx])

        return total_sum/count

    # Returns the current statistics of profiled lifetimes
    def get_details(self) -> dict:
        # Do calculations here and send
        return {
            'mean_lifetimes': self.__mean,
            'most_recent': self.__most_recently_used
            }

# Greedy retrival thread
class GreedyRetrievalThread (threading.Thread):
    def __init__(self, thread_id, name, caller):
        threading.Thread.__init__(self)
        self.thread_id = thread_id 
        self.caller = caller # Pointer to the caller instance
        self.name = name

    def run(self):
        self.refresh()

    def refresh(self):
        while True:
            # Trigger refresh event
            post_event("need_to_refresh", self.name)
            
            # Get the currently inffered lifetime of the attribute
            means = getattr(self.caller,'mean')
            # Sleep until that time is elapsed
            time.sleep(means[self.thread_id]/1000)
            