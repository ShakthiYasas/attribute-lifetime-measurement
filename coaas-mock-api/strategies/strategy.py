# Abstract class
class Strategy(object):
    """Base class of all the strategies"""
    
    service_registry = None
    cache_memory = None
    attributes = 0
    session = ''

    def __init__(self, db, window, isstatic=True): pass
    def init_cache(self): pass
    def get_result(self, json = None, fthr = 0, session = None) -> dict: pass
    def get_current_profile(self): pass
    
