import matplotlib.pyplot as plt
from configuration import CarParkConfiguration
from distribution import NormalDistribution, RandomDistribution, SuperImposedDistribution

class CarPark:
    configuration = None
    distribution = []
    
    def __init__(self,configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            self.configuration = configuration
        else:
            print('Provided configuration is invalid. Proceeding with default!')
            self.configuration = CarParkConfiguration()

        print('Initializing Carpark')
        for count in range(0,len(self.configuration.skew)):
            if(self.configuration.variation[count] == 0):
                self.distribution.append(RandomDistribution(self.configuration, count))
            elif(abs(self.configuration.variation[count]) == 1):
                self.distribution.append(NormalDistribution(self.configuration, count))
            else:
                self.distribution.append(SuperImposedDistribution(self.configuration, count))

        self.print_distrubtions()
        print('Car park service running!')


    def get_current_status(self, milisecond_diff) -> dict:
        current_time_step = milisecond_diff/self.configuration.sampling_rate
        response_obj = dict()
        for idx in range(0,len(self.distribution)):
            response_obj['area_'+str(idx+1)+'_availability'] = self.distribution[idx].get_occupancy_level(current_time_step)

        return response_obj
        
    def print_distrubtions(self):
        plt.xlabel('Time Step')
        plt.ylabel('occupancy')
        
        for dist in self.distribution:
            idx = self.distribution.index(dist)
            if(self.configuration.variation[idx] < 0):
                dist.occupancy = list(map(lambda x : self.configuration.sample_size - x, dist.occupancy))

            plt.plot(dist.time_step, dist.occupancy)
            
            f = open(str(self.configuration.current_session)+'-simulation-area_'+str(self.distribution.index(dist)+1)+'_availability', "a")
            f.write('time_step,occupancy\n')
            for idx in range(0,len(dist.time_step)):
                f.write(str(dist.time_step[idx]*self.configuration.sampling_rate)+','+str(dist.occupancy[idx])+'\n')
            f.close()
  
        plt.savefig(str(self.configuration.current_session)+'-distribution.png')

