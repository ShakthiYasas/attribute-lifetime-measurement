from cache.eviction.lfuevictor import LFUEvictor
from cache.eviction.tlruevictor import TLRUEvictor
from cache.eviction.lvfevictor import LVFEvictor
from cache.eviction.novelevictor import NovelEvictor
from cache.eviction.ranodmevictor import RandomEvictor

class EvictorFactory:
    __evictor = None
    def __init__(self, evictalgo):
        if self.__evictor is None:
            if(evictalgo == 'lfu'):
                print('Initializing Least-Frequently-Used eviction for the cache memory.')
                self.__evictor = LFUEvictor()
            elif(evictalgo == 'tlru'):
                print('Initializing Time-aware Least Recently Used eviction for the cache memory.')
                self.__evictor = TLRUEvictor()
            elif(evictalgo == 'lvf'):
                print('Initializing Least-Valued-First eviction for the cache memory.')
                self.__evictor = LVFEvictor()
            elif(evictalgo == 'novel'):
                print('Initializing our novel eviction strategy for the cache memory.')
                self.__evictor = NovelEvictor()
            else:
                print('Initializing Random eviction for the cache memory.')
                self.__evictor = RandomEvictor()

    def getevictor(self):
        return self.__evictor
