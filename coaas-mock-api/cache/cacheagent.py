# Abstract class
class CacheAgent(object):
    def __init__(self, size): pass
    # Eviction
    def evict(self, entityid) -> None: pass
    # Store 
    def save(self, entityid, cacheitems) -> None: pass
    # Retrieve
    def get_values(self) -> dict: pass
    def get_value_by_key(self,entityid,attribute): pass
    # Check Cache
    def _is_cached(self,entityid,attribute): pass
    # Get Statistics
    def get_statistics(self): pass
    def get_statistics(self, entityid, attribute): pass
