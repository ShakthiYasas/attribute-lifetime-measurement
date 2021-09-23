import datetime
from math import trunc
from dateutil import parser
from profiler import Profiler
from restapiwrapper import Requester
from strategies.strategy import Strategy

class Adaptive(Strategy):   
    db_insatnce = None

    def __init__(self, attributes, url, db, window):
        print('Initializing Adaptive Profile') 
        self.url = url
        self.meta = None
        self.db_insatnce = db
        self.moving_window = window
        self.requester = Requester()
        self.profiler = Profiler(attributes, db, self.moving_window, self.__class__.__name__.lower())
    
    def init_cache(self):
        self.profiler.session = self.session
        response = self.requester.get_response(self.url)
        
        self.meta = response['meta']
        self.meta['start_time'] = parser.parse(self.meta['start_time'])
        
        del response['meta']
        time_diff = (datetime.datetime.now() - self.meta['start_time']).total_seconds()*1000
        response['step'] = trunc(time_diff/self.meta['sampling_rate'])

        self.profiler.reactive_push(response)
        self.cache_memory.save(response)

    def get_result(self, url = None, json = None, session = None):               
        query = []
        refetching = []
        now = datetime.datetime.now()

        if(len(json) != 0):
            # check freshness of the given attributes
            for item in json:
                if(self.cache_memory.get_value_by_key(item['attribute']) != None):
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

        time_diff = (now - self.meta['start_time']).total_seconds()*1000
        output = {'step': trunc(time_diff/self.meta['sampling_rate'])}

        if(len(json) != 0):       
            for item in json:
                cached_item  = self.cache_memory.get_value_by_key(item['attribute'])
                if(cached_item != None):
                    output[item['attribute']] = cached_item

        self.db_insatnce.insert_many('adaptive-hits', query)
        
        return output

    def get_current_profile(self):
        self.profiler.get_details()

    def refresh_cache(self, attributes) -> None:
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

        self.profiler.reactive_push(response)
        self.cache_memory.save(response)

        #run_in_parallel(self.cache_memory.save(response),self.profiler.reactive_push(response))