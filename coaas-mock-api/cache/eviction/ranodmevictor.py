import random
import numpy as np

from cache.eviction.evictor import Evictor

# How many attributes and entities to evict?
# When to evict?
class RandomEvictor(Evictor):
    def __init__(self, parent_cache, threshold = 1.0):
        self.__cache = parent_cache

    def select_for_evict(self):
        entityids = list(self.__cache.get_statistics_all().keys())
        selected_entity = entityids[np.random.randint(0,len(entityids))]

        attributes = list(self.__cache.get_statistics_entity(selected_entity).keys())
        selected_att = attributes[np.random.randint(0,len(attributes))]

        return [(selected_entity, selected_att)]

    # Select a random entity to evict
    def select_entity_to_evict(self, internal=False, is_limited=False):
        secure_random = random.SystemRandom()
        if(is_limited):
            return [secure_random.choice([entityid for entityid in self.__cache.get_statistics_all().keys()])]
        return [secure_random.choice([entityid for entityid in self.__cache.get_statistics_all().keys()])]