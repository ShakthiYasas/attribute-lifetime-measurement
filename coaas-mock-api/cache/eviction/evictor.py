# Abstract Class
class Evictor(object):
    def __init__(self, cache, threshold = 1.0): pass
    def select_for_evict(self): pass
    def select_entity_to_evict(self, internal=False): pass