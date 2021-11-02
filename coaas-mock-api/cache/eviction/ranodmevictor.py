import numpy as np

from cache.eviction.evictor import Evictor
from cache.inmemcache import InMemoryCache

# How many attributes and entities to evict?
# When to evict?
class RandomEvictor(Evictor):
    def __init__(self, parent_cache:InMemoryCache):
        self.__cache = parent_cache

    def select_for_evict(self):
        entityids = self.__cache.get_statistics_all().keys()
        selected_entity = entityids[np.random.randint(0,len(entityids)+1)]

        attributes = self.__cache.get_statistics_entity(selected_entity).keys()
        selected_att = attributes[np.random.randint(0,len(attributes)+1)]

        return [(selected_entity, selected_att)]