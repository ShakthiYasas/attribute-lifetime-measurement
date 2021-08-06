from configuration import CarParkConfiguration
from distribution import NormalDistribution, RandomDistribution

class CarPark:
    configuration = None
    distribution = []

    def __init__(self):
        print('Initializing Carpark')
        self.configuration = CarParkConfiguration()
        for count in range(0,len(self.configuration.skew)):
            self.distribution.append(NormalDistribution(self.configuration, count))
    
    def __init__(self,configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            self.configuration = configuration
        else:
            print('Provided configuration is invalid. Proceeding with default!')
            self.configuration = CarParkConfiguration()

        if(self.configuration.variation != 0):
            for count in range(0,len(self.configuration.skew)):
                self.distribution.append(NormalDistribution(self.configuration, count))
        else:
            for count in range(0,len(self.configuration.skew)):
                self.distribution.append(RandomDistribution(self.configuration, count))

    def get_current_status(self, milisecond_diff) -> dict:
        current_time_step = milisecond_diff/self.configuration.sampling_rate
        response_obj = dict()
        for idx in range(0,len(self.distribution)):
            response_obj['area_'+str(idx+1)+'_availability'] = self.distribution[idx].get_occupancy_level(current_time_step)

        return response_obj
        
    
