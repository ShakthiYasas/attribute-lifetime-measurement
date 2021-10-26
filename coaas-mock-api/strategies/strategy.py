# Abstract class
class Strategy:
    """Base class of all the strategies"""
    cache_memory = None
    attributes = 0
    session = ''

    def __init__(self, db, window): pass
    def init_cache(self): pass
    def get_result(self, json = None, session = None) -> dict: pass
    def get_current_profile(self): pass
    
