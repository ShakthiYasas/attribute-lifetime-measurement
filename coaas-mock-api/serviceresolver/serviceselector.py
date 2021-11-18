import threading
import statistics
from datetime import datetime, timedelta
from lib.restapiwrapper import Requester
from lib.fifoqueue import FIFOQueue_2

# Simple Context Service Resolution
class ServiceSelector:
    __statistics = dict()
    __statsLock = threading.Lock()

    def __init__(self, db):
        self.requester = Requester()
        self.__db = db

    def get_response_for_entity(self, attributes:list, urllist:list):
        output = {}
        now = datetime.now()
        threads = []
        for prodid, url in urllist:
            t = threading.Thread(target=self.__get_context(prodid, url, attributes, now, output))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        return output

    def __get_context(self, prodid, url, attributes, now, output):
        res = self.requester.get_response(url)
        if(res):
            aft_time = datetime.now()
            responetime = (aft_time-now).total_seconds()

            if(not (prodid in self.__statistics)):
                queue = FIFOQueue_2(100)
                self.__statistics[prodid] = {'count': 1}
                queue.push(responetime)
                self.__statistics[prodid]['queue'] = queue
            else:
                self.__statsLock.acquire()
                self.__statistics[prodid]['count'] += 1
                self.__statistics[prodid]['queue'].push(responetime)
                self.__statsLock.release()

            if(self.__statistics[prodid]['count'] % 5 == 0):
                self.__db.insert_one('responsetimes',{
                    'context_producer': prodid,
                    'avg_response_time': statistics.mean(self.__statistics[prodid]['queue'].getlist()),
                    'timestamp': aft_time
                })

            for att in attributes:
                if(not (att in output) and att in res):
                    wait_time = now - timedelta(seconds=responetime)
                    output[att] = [(prodid,res[att],wait_time)]
                elif(att in res):
                    self.__statsLock.acquire()
                    wait_time = now - timedelta(seconds=responetime)
                    output[att].append((prodid,res[att], wait_time))
                    self.__statsLock.release()
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
            return None