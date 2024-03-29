import time
import _thread
import threading
import statistics
from datetime import datetime, timedelta
from lib.restapiwrapper import Requester
from lib.fifoqueue import FIFOQueue_2

# Simple Context Service Resolution
class ServiceSelector:
    session = ''
    service_registry = None

    __statistics = dict()
    __stats_lock = threading.Lock()
    __recent_history = FIFOQueue_2(100)
    
    __provider_hash = {}
    __hash_lock = threading.Lock()

    def __init__(self, db, window = 5000):
        self.requester = Requester()
        self.__db = db
        self.__window = window

        # Initializing background thread to clear hash
        thread = threading.Thread(target=self.__run, args=())
        thread.daemon = True               
        thread.start() 

    def get_response_for_entity(self, attributes:list, urllist:list):
        output = {}
        now = datetime.now()

        for prodid, url in urllist:
            _thread.start_new_thread(self.__update_recency_bit, (prodid,))
            self.__get_context(prodid, url, attributes, now, output)

        return output

    def __run(self):
        while True:
            self.__reset_recency_bit()
            time.sleep(self.__window/1000)

    def __update_recency_bit(self, prodid):
            self.__hash_lock.acquire()
            self.__provider_hash[prodid] = True
            self.__hash_lock.release()

    def __reset_recency_bit(self):
        self.__provider_hash = {rec[0]: False for rec in self.__provider_hash.items()}
        
    def get_current_retrival_latency(self):
        lst = self.__recent_history.getlist()
        return statistics.mean(lst) if len(lst)>0 else 0

    def __get_context(self, prodid, url, attributes, now, output): 
        meta_data = self.service_registry.get_provider_meta(prodid, attributes)
        if(meta_data != None):
            for att in attributes:
                if(att in meta_data.keys()):
                    if(not (att in output)):
                        output[att] = [(prodid,meta_data[att],now)]
                    else:
                        self.__stats_lock.acquire()
                        output[att].append((prodid,meta_data[att],now))
                        self.__stats_lock.release()
            attributes = set(attributes) - set(meta_data.keys())

        if(attributes):
            res = self.requester.get_response(url) 
            if(res):
                aft_time = datetime.now()
                responetime = (aft_time-now).total_seconds()
                self.__recent_history.push(responetime)

                if(not (prodid in self.__statistics)):
                    queue = FIFOQueue_2(100)
                    self.__statistics[prodid] = {'count': 1}
                    queue.push(responetime)
                    self.__statistics[prodid]['queue'] = queue
                else:
                    self.__stats_lock.acquire()
                    self.__statistics[prodid]['count'] += 1
                    self.__statistics[prodid]['queue'].push(responetime)
                    self.__stats_lock.release()

                if(self.__statistics[prodid]['count'] % 5 == 0):
                    self.__db.insert_one('responsetimes',{
                        'context_producer': prodid,
                        'avg_response_time': statistics.mean(self.__statistics[prodid]['queue'].getlist()),
                        'cum_retrievals': self.__statistics[prodid]['count'],
                        'timestamp': aft_time,
                        'session':self.session
                    })

                for att in attributes:
                    if(not (att in output) and att in res):
                        response_latency = now - timedelta(seconds=responetime)
                        output[att] = [(prodid,res[att],response_latency)]
                    elif(att in res):
                        self.__stats_lock.acquire()
                        response_latency = now - timedelta(seconds=responetime)
                        output[att].append((prodid,res[att], response_latency))
                        self.__stats_lock.release()
            else:
                # This producer is either invalid or currently having issues
                # Could be skipped for a few retrievals
                print('Context Provider is Invalid or Unavilable.')

    def get_average_responsetime_for_provider(self, prodid):
        if(prodid in self.__statistics):
            rts = self.__statistics[prodid]['queue'].getlist()
            return sum(rts)/len(rts)
        return None
    
    def get_average_responsetime_for_attribute(self, contextprodiverids):
        rtlist = []
        for cp in contextprodiverids:
            rt = self.get_average_responsetime_for_provider(cp)
            if(rt is None):
                res = self.__db.read_all_with_limit('responsetimes',{
                    'context_producer': cp
                },10)
                if(res):
                    for values in res:
                        rtlist.append(values['avg_response_time'])
            else:
                rtlist.append(rt)

        if(rtlist):
            return statistics.mean(rtlist)
        else:
            return 9999