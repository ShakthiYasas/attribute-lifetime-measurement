#Abstract Class
class Profiler(object):
    """Base class for the 2 profiler classes"""
    def __init__(self, attributes, db, window, caller_name, session = None): pass
    def reactive_push(self, response, is_greedy=False) -> None: pass
    def clear_expired(self) -> None: pass
    def get_details(self) -> dict: pass