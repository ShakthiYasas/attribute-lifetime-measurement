from configurations.configuration import Configuration

# Configuration of the Cache Memory
class CacheConfiguration(Configuration):
    # Default Configuration
    cache_size = 50
    window_size=10000
    type = 'in-memory'
    
    def __init__(self, config):
        self.type = config['CacheType']
        self.window_size = int(config['WindowSize'])
        self.eviction_algo = config['EvictionAlgo']
        self.cache_size = int(config['CacheBlocksPerUnit'])*int(config['CacheSize'])

        
