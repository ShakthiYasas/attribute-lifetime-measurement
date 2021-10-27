from datetime import datetime
from lib.restapiwrapper import Requester
from lib.fifoqueue import FIFOQueue

# Simple Context Service Resolution
class ServiceSelector:
    def __init__(self):
        self.requester = Requester()
        self.__statistics = {}

    def get_response_for_entity(self, attributes:list, urllist:list):
        output = {}
        for prodid, url in urllist:
            now = datetime.now()
            res = self.requester.get_response(url)
            aft_time = datetime.now()
            if(prodid in self.__statistics):
                self.__statistics[prodid] = FIFOQueue().push(aft_time-now)
            else:
                self.__statistics[prodid].push(aft_time-now)
            for att in attributes:
                if(att in output):
                    output[att] = [(prodid,res[att],now)]
                else:
                    output[att].append((prodid,res[att],now))
        return output

    def get_average_responsetime_for_provider(self, prodid):
        if(prodid in self.__statistics):
            rts = self.__statistics[prodid].getlist()
            return sum(rts)/len(rts)
        return None