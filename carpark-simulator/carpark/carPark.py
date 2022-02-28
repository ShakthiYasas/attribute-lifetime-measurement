from math import trunc
import matplotlib.pyplot as plt
from configuration import CarParkConfiguration
from distribution import NormalDistribution, RandomDistribution, SuperImposedDistribution, LinearDistribution, StaticDistribution

class CarPark:
    # Class variables
    configuration = None
    distribution = []
    
    def __init__(self,configuration):
        if(isinstance(configuration,CarParkConfiguration)):
            self.configuration = configuration
        else:
            print('Provided configuration is invalid. Proceeding with default!')
            self.configuration = CarParkConfiguration()

        print('Initializing Carpark')
        # Selecting the correct type occupancy data distbution for the car park
        # based on configuration.
        for count in range(0,len(self.configuration.skew)):
            if(self.configuration.variation[count] == 0):
                self.distribution.append(RandomDistribution(self.configuration, count, configuration.random_noise))
            elif(abs(self.configuration.variation[count]) == 1):
                self.distribution.append(NormalDistribution(self.configuration, count, configuration.random_noise))
            elif(abs(self.configuration.variation[count]) == 2):
                self.distribution.append(SuperImposedDistribution(self.configuration, count, configuration.random_noise))
            elif(self.configuration.variation[count] == 5):
                self.distribution.append(LinearDistribution(self.configuration, configuration.random_noise))
            elif(self.configuration.variation[count] == 6):
                self.distribution.append(StaticDistribution(self.configuration, configuration.random_noise))

        self.print_distrubtions()
        print('Car park service running!')

    # Produces the current occupancy
    def get_current_status(self, milisecond_diff) -> dict:
        current_time_step = trunc(milisecond_diff/self.configuration.sampling_rate)
        response_obj = dict()
        for idx in range(0,len(self.distribution)):
            response_obj['area_'+str(idx+1)+'_availability'] = self.distribution[idx].get_occupancy_level(current_time_step)

        return response_obj
        
    # Generates a, 
    #  - Graphical visualization of the generated data distribution
    #  - CSV file of the discrete lifetimes
    # The files are saved within the container (or the directory)
    def print_distrubtions(self):
        plt.xlabel('Time Step')
        plt.ylabel('occupancy')
        
        for dist in self.distribution:
            idx = self.distribution.index(dist)
            if(self.configuration.variation[idx] < 0):
                dist.occupancy = list(map(lambda x : self.configuration.sample_size - x, dist.occupancy))

            plt.plot(dist.time_step, dist.occupancy)
            
            # Saving the CSV file
            life_file = open(str(self.configuration.current_session)+'-simulation-area_'+str(self.distribution.index(dist)+1)+'-lifetimes.csv', "a")
            life_file.write('start,end,occupancy,life\n')

            start, end = 0, 0
            curr_value = dist.occupancy[0]
            for idx in range(0,len(dist.time_step)):
                end = idx
                if(start != end and curr_value != dist.occupancy[end]):
                    life = (dist.time_step[end] - dist.time_step[start])*self.configuration.sampling_rate
                    life_file.write(str(dist.time_step[start])+','+str(dist.time_step[end])+','+str(dist.occupancy[idx-1])+','+str(life)+'\n')
                    start = end
                    curr_value = dist.occupancy[end]
                elif(idx == len(dist.time_step)-1):
                    life = (dist.time_step[end] - dist.time_step[start])*self.configuration.sampling_rate
                    life_file.write(str(dist.time_step[start])+','+str(dist.time_step[end])+','+str(dist.occupancy[idx])+','+str(life)+'\n')

            life_file.close()

        # Generating the graphical representation
        plt.savefig(str(self.configuration.current_session)+'-distribution.png')

