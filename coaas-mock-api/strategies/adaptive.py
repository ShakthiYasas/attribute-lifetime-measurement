import datetime
from strategies.strategy import Strategy
from profiler import Profiler
from restapiwrapper import Requester
#from lib.util import run_in_parallel

class Adaptive(Strategy):   
    db_insatnce = None

    def __init__(self, attributes, url, db, window):
        print('Initializing Adaptive Profile')
        self.url = url
        self.db_insatnce = db
        self.moving_window = window
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())
    
    def init_cache(self):
        self.profiler.session = self.session
        response = self.requester.get_response(self.url)
        self.profiler.reactive_push(response)
        self.cache_memory.save(response)

    def get_result(self, url = None, json = None, session = None):       
        response = self.cache_memory.get_values()
        now = datetime.datetime.now()

        refetching = []
        query = []
        if(len(json) != 0):
            # check freshness of the given attributes
            for item in json:
                if(item['attribute'] in response):
                    idx = self.profiler.lookup[item['attribute']]
                    l_f = self.profiler.most_recently_used[idx]
                    last_fecth = self.profiler.last_time if not l_f else l_f[-1][1]
                    mean_for_att = self.profiler.mean[idx]
                    expire_time = mean_for_att * (1 - item['freshness'])
                    time_at_expire = last_fecth + datetime.timedelta(milliseconds=expire_time)
                    if(now > time_at_expire):
                        refetching.append(item['attribute'])
                        query.append({'session': session, 'attribute': item['attribute'], 'isHit': False})
                    else:
                        query.append({'session': session, 'attribute': item['attribute'], 'isHit': True})
        else:
            # check freshness of all attributes 
            for item in json:
                idx = self.profiler.lookup[item['attribute']]
                l_f = self.profiler.most_recently_used[idx]
                last_fecth = self.profiler.last_time if not l_f else l_f[-1][1]
                mean_for_att = self.profiler.mean[idx]
                expire_time = mean_for_att * (1 - item['freshness'])
                time_at_expire = last_fecth + datetime.timedelta(milliseconds=expire_time)
                if(now > time_at_expire):
                    refetching.append(item['attribute'])
                    query.append({'session': session, 'attribute': item['attribute'], 'isHit': False})
                else:
                    query.append({'session': session, 'attribute': item['attribute'], 'isHit': True})

        self.refresh_cache(refetching)
        response = self.cache_memory.get_values()

        if(len(json) != 0):
            modified_response = {}
            for item in json:
                if(item['attribute'] in response):
                    modified_response[item['attribute']] = response[item['attribute']]
            response = modified_response

        self.db_insatnce.insert_many('adaptive-hits', query)
        return response

    def get_current_profile(self):
        self.profiler.get_details()

    def refresh_cache(self, attributes) -> None:
        response = self.requester.get_response(self.url)
        
        modified_response = {}
        for att in attributes:
            if(att in response):
                modified_response[att] = response[att]
        response = modified_response

        self.profiler.reactive_push(response)
        self.cache_memory.save(response)

        #run_in_parallel(self.cache_memory.save(response),self.profiler.reactive_push(response))