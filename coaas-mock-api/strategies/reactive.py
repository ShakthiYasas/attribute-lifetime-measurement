import time
import _thread
import threading
import statistics

from lib.fifoqueue import FIFOQueue_2
from strategies.strategy import Strategy
from profilers.adaptiveprofiler import AdaptiveProfiler
from serviceresolver.serviceselector import ServiceSelector

# Reactive retrieval strategy
# This strategy would retrieve from the context provider for all the requests.
# Therefore, this is only a redirector. No cache involved.
# Reactive is a benchmark for the other 2 stragies since, the highest freshness can be achieved (Data accuracy).
# However, expecting the lowest response time.

class Reactive(Strategy):
    # Counters
    __window_counter = 0

    # Queues
    __sla_trend = FIFOQueue_2(10)
    __request_rate_trend = FIFOQueue_2(1000)
    __retrieval_cost_trend = FIFOQueue_2(10)
    
    def __init__(self, db, window, isstatic=True, learncycle = 20, skip_random=False):
        self.__db = db
        self.__reqs_in_window = 0
        self.__isstatic = isstatic 
        self.__local_ret_costs = []
        self.__moving_window = window
        self.__most_expensive_sla = (0.5,1.0,1.0)

        self.service_selector = ServiceSelector(db)
        if(not self.__isstatic):
            self.__profiler = AdaptiveProfiler(db, self.__moving_window, self.__class__.__name__.lower())  
 
    # Init_cache initializes the cache memory. 
    def init_cache(self):
        setattr(self.service_selector, 'service_registry', self.service_registry)

        # Set current session to profiler if not set
        if((not self.__isstatic) and self.__profiler and self.__profiler.session == None):
            self.__profiler.session = self.session

        # Initializing background thread clear observations.
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True               
        thread.start() 
    
    def run(self):
        while True:
            self.clear_expired()
            _thread.start_new_thread(self.__save_current_cost, ())
            self.__window_counter+=1
            time.sleep(self.__moving_window/1000) 

    # Clear function that run on the background
    def clear_expired(self) -> None:        
        _thread.start_new_thread(self.service_registry.update_ret_latency, (self.service_selector.get_current_retrival_latency(),))
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

    # Save cost variation
    def __save_current_cost(self):
        self.__db.insert_one('returnofcaching', {
            'session':self.session,
            'window': self.__window_counter - 1,
            'return': self.get_current_cost()
        })
    
    def get_current_cost(self):
        # Hit rate is always 0 because there is not cache
        sla = self.__sla_trend.get_last()
        request_rate = self.__request_rate_trend.get_last()
        ret_cost = self.__retrieval_cost_trend.get_last()
        return -request_rate*(sla[2] + ret_cost)

    # Returns the current statistics from the profiler
    def get_current_profile(self):
        self.__profiler.get_details()

    # Returns the variation of average cost of context caching
    def get_cost_variation(self, session = None):
        output = {}      
        if(session):
            stats = self.__db.read_all('returnofcaching', {'session': session})
            if(stats):
                for stat in stats:
                    output[stat['window']] = stat['return']
        else:
            slas = self.__sla_trend.getlist()
            request_rates = self.__request_rate_trend.get_last_range(10)
            ret_costs = self.__retrieval_cost_trend.get_last_range(10)
            for i in range(0,10):
                idx = -1-i
                try:
                    sla = slas[idx]
                    ret_cost = ret_costs[idx]
                    output[idx] = request_rates[idx]*(sla[2]+ret_cost)*(-1)
                except Exception:
                    output[idx] = 0

        return output
    
    # Get the low-level context requested in the query
    def get_result(self, json = None, fthresh = (0.5,1.0,1.0), req_id = None):   
        # Set current session to profiler if not set
        if((not self.__isstatic) and self.__profiler and self.__profiler.session == None):
            self.__profiler.session = self.session
        
        # Retrieve context data directly from provider 
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

        output = {}
        for ent in json:
            entityid = ent['entityId']
            conditions = []
            if('conditions' in ent):
                conditions = ent['conditions']
            lifetimes = None
            if(self.__isstatic):
                lifetimes = self.service_registry.get_context_producers(entityid,ent['attributes'],conditions)

            if(lifetimes):
                    out = self.__retrieve_entity(ent['attributes'],lifetimes)
                    output[entityid] = {}
                    for att_name,prod_values in out.items():
                        output[entityid][att_name] = [res[1] for res in prod_values]

        return output
                        
    # Retrieving context for an entity
    def __retrieve_entity(self, attribute_list: list, metadata: dict) ->  dict:
        # Retrive raw context from provider according to the entity
        return self.service_selector.get_response_for_entity(attribute_list, 
                    list(map(lambda k: (k[0],k[1]['url']), metadata.items())))
    
