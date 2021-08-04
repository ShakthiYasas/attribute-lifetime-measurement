from configuration import CarParkConfiguration
from distribution import NormalDistribution, RandomDistribution

class CarPark:
    configuration = None
    distribution = None

    def __init__(self):
        print('Initializing Carpark')
        self.configuration = CarParkConfiguration()
        self.distribution = NormalDistribution(self.configuration)
    
    def __init__(self,configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            self.configuration = configuration
        else:
            print('Provided configuration is invalid. Proceeding with default!')
            self.configuration = CarParkConfiguration()

        if(self.configuration.variation != 0):
            self.distribution = NormalDistribution(self.configuration)
        else:
            self.distribution = RandomDistribution(self.configuration)

    def get_current_status(self, milisecond_diff) -> dict:
        current_time_step = milisecond_diff/self.configuration.sampling_rate
        return {'current_availability': self.distribution.get_occupancy_level(current_time_step)} 
        
    
