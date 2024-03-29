# Abstract class
class Strategy(object):
    """Base class of all the strategies"""
    selective_agent_factory = None
    selective_cache_agent = None
    service_registry = None
    cache_memory = None
    trend_ranges = []
    attributes = 0
    session = ''

    def __init__(self, db, window, isstatic=True, learncycle = 20): pass
    def init_cache(self): pass
    def get_result(self, json = None, fthr = (0,1.0,1.0), session = None) -> dict: pass
    def get_current_profile(self): pass
    
