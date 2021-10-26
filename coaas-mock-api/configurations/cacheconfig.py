from configurations.configuration import Configuration

# Configuration of the Cache Memory
class CacheConfiguration(Configuration):
    # Default Configuration
    cache_size = 50
    window_size=10000
    type = 'in-memory'
    
    def __init__(self, config):
        defaults = config['DEFAULT']
       
        self.type = defaults['CacheType']
        self.window_size = defaults['WindowSize']
        self.cache_size = int(defaults['CacheBlocksPerUnit'])*int(defaults['CacheSize'])

        
