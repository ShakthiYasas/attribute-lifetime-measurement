from datetime import date
from lib.restapiwrapper import Requester
from datetime import datetime

# Simple Context Service Resolution
class ServiceSelector:
    def __init__(self):
        self.requester = Requester()

    def get_response_for_entity(self, attributes:list, urllist:list):
        output = {}
        now = datetime.now()
        for prodid, url in urllist:
            res = self.requester.get_response(url)
            for att in attributes:
                if(att in output):
                    output[att] = [(prodid,res[att],now)]
                else:
                    output[att].append((prodid,res[att],now))
        return output