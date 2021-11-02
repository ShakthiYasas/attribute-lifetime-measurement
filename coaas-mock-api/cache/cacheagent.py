from lib.limitedsizedict import LimitedSizeDict
# Abstract class
class CacheAgent(object):
    def __init__(self, config, db, registry=None): pass
    # Eviction
    def evict(self, entityid) -> None: pass
    # Store 
    def save(self, entityid, cacheitems) -> None: pass
    # Retrieve
    def get_values(self) -> LimitedSizeDict: pass
    def get_value_by_key(self,entityid,attribute): pass
    def get_values_for_entity(self,entityid,attr_list): pass
    # Check Cache (private method)
    def __is_cached(self,entityid,attribute): pass
    # Get Statistics
    def get_statistics(self): pass
    def get_hitrate_trend(self): pass
    def get_statistics(self, entityid, attribute): pass
