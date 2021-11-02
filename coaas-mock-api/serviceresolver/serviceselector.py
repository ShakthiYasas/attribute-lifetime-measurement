import statistics
from datetime import datetime
from lib.restapiwrapper import Requester
from lib.fifoqueue import FIFOQueue

# Simple Context Service Resolution
class ServiceSelector:
    def __init__(self, db):
        self.requester = Requester()
        self.__statistics = {}
        self.__db = db

    def get_response_for_entity(self, attributes:list, urllist:list):
        output = {}
        for prodid, url in urllist:
            now = datetime.now()
            res = self.requester.get_response(url)
            aft_time = datetime.now()
            if(prodid in self.__statistics):
                self.__statistics[prodid] = {
                    'count': 1,
                    'queue': FIFOQueue().push((aft_time-now).total_seconds())
                }
            else:
                self.__statistics[prodid]['count']+=1
                self.__statistics[prodid]['queue'].push((aft_time-now).total_seconds())

            if(self.__statistics[prodid]['count'] % 5):
                self.__db.insert_one('responsetimes',{
                    'context_producer': prodid,
                    'avg_response_time': statistics.mean(self.__statistics[prodid]['queue'].getlist()),
                    'timestamp': aft_time
                })

            for att in attributes:
                if(att in output):
                    output[att] = [(prodid,res[att],now)]
                else:
                    output[att].append((prodid,res[att],now))
        return output

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
                    local_rt = list(map(lambda x: x['avg_response_time'], res))
                    rtlist.append(statistics.mean(local_rt))
            else:
                rtlist.append(rt)

        if(rtlist):
            return statistics.mean(rtlist)
        else:
            return None