from strategy import Strategy
from restapiwrapper import Requester

class Reactive(Strategy):
    def __init__(self):
        self.requester = Requester()

    def get_result(self, url):       
        self.requester.get_response(url)
        
