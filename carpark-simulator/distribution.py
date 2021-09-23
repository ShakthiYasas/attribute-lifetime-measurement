import random
import numpy as np
from matplotlib import mlab
from math import trunc
from scipy.stats import skewnorm
from scipy.ndimage import gaussian_filter1d

from error import ValueConfigurationError
from configuration import CarParkConfiguration

class Distribution(object):
    """"Base class of all distributions"""
    pass

# NormalDistribution class 
# Generates a standard normal distribution that is guranteed skewed.
# Config Value: up, down
class NormalDistribution(Distribution):
    def __init__(self, configuration, index, noise=False):
        # Class Variables
        self.time_step = []
        self.occupancy = []
        
        self.sample_size = configuration.sample_size
        self.seletced_periods = configuration.selected_periods
        self.variation = configuration.variation[index]

        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a skewed normal distribution.')

            # Generate a random distribution
            linspace = np.linspace(start=0, stop=configuration.sample_size, num=configuration.no_of_refreshes)
            init_dist = skewnorm.pdf(linspace, a=configuration.skew[index], scale=configuration.standard_deviation[index])
        
            # Normalize the distribution 
            distributions = normalizing_distribution(init_dist, configuration.sample_size)
            self.time_step = distributions[0]

            # Smoothening the curve
            self.occupancy = smooth(distributions[1],configuration.standard_deviation[index])

            # Discretizing the distribtution into time steps
            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods,self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]
            
            # Add random noise to the distribution if set True
            if(noise):
                self.occupancy = randomnoiser(self.occupancy, configuration.sample_size, configuration.noise_percentage)

            print('Random skewed distribution created.')
        else:
            raise ValueConfigurationError()

    # Get the current occupnacy of the car-park
    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)

        # Mapping continuous time into discrete timesteps 
        if(closest_step>step):
            if(self.variation>0):
                return self.occupancy[idx-1]
            else:
                return self.sample_size - self.occupancy[idx-1]
        else:
            if(self.variation>0):
                return self.occupancy[idx]
            else:
                return self.sample_size - self.occupancy[idx]

# RandomDistribution class 
# Generates a standard normal distribution that is guranteed no-skew.
# Config Value: random
class RandomDistribution(Distribution):
    def __init__(self, configuration, index, noise=False):
        print('Starting to create a random normal distribution.')

        # Class variable
        self.time_step = []
        self.occupancy = []
        self.sample_size = configuration.sample_size
        self.variation = configuration.variation[index]

        number_of_data = (configuration.sample_size)**2
        init_dist = np.random.normal(0, configuration.standard_deviation[index], number_of_data)
        sorted_array = np.sort(init_dist)

        min_value = sorted_array[0]
        max_value = sorted_array[-1]
        diff = max_value - min_value
        step_size = diff/configuration.no_of_refreshes

        current_frequency = 0
        current_index = 0
        try:
            for snap in range(1,configuration.no_of_refreshes+1):
                current_upbound = min_value + (snap*step_size)
                for idx in range(current_index,number_of_data):
                    if(sorted_array[idx]<=current_upbound):
                        current_frequency += 1
                    else:
                        self.time_step.append(snap)
                        self.occupancy.append(current_frequency)
                        current_frequency = 0
                        current_index = idx
                        break

            # Discretizing the distribtution into time steps
            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods, self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]

            # Smoothing the curve
            self.occupancy = smooth(self.occupancy, configuration.standard_deviation[index])
            
        except(Exception):
            print('End of distribution')

        self.occupancy = list(map(lambda x: configuration.sample_size if x > configuration.sample_size else x, 
                                smooth(self.occupancy, configuration.standard_deviation[index])))
        
        # Add random noise to the distribution if set True
        if(noise):
                self.occupancy = randomnoiser(self.occupancy, self.sample_size, configuration.noise_percentage)

    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)
        
        # Mapping continuous time into discrete timesteps 
        if(closest_step>step):
            if(self.variation>0):
                return self.occupancy[idx-1]
            else:
                return self.sample_size - self.occupancy[idx-1]
        else:
            if(self.variation>0):
                return self.occupancy[idx]
            else:
                return self.sample_size - self.occupancy[idx]

# SuperImposedDistribution class
# Superimposes skewed normal distribution with a un-skewed normal distibution to produce
# a shifted normal distibution, having higher variance.
# Config Value: super-up, super-down
class SuperImposedDistribution(Distribution):
        def __init__(self, configuration, index, noise=False):  
            # Class variables
            self.time_step = []
            self.occupancy = []
            
            self.sample_size = configuration.sample_size
            self.variation = configuration.variation[index]

            # Creating thhe baseline distributions
            random = RandomDistribution(configuration, index)
            norm = NormalDistribution(configuration, index)

            res = super_impose(random, norm)
            self.time_step = res[0]
            self.occupancy = res[1]

            # Add random noise to the distribution if set True
            if(noise):
                self.occupancy = randomnoiser(self.occupancy, self.sample_size, configuration.noise_percentage)

        # Get current occupnacy of the carpark
        def get_occupancy_level(self, step):
            closest_step = min(self.time_step, key=lambda x:abs(x-step))
            idx = self.time_step.index(closest_step)

            # Mapping continuous time into discrete timesteps 
            if(closest_step>step):
                if(self.variation>0):
                    return self.occupancy[idx-1]
                else:
                    return self.sample_size - self.occupancy[idx-1]
            else:
                if(self.variation>0):
                    return self.occupancy[idx]
                else:
                    return self.sample_size - self.occupancy[idx]

# StaticDistribution class
# Generates a linear distribution where gradient = 0 (i.e. y=n where n is real number)
# Config Value: static-line
class StaticDistribution(Distribution):
    def __init__(self, configuration, noise = False):
        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a static linear distribution.')
            self.time_step = range(0, configuration.no_of_refreshes)
            self.occupancy = [configuration.min_occupancy] * configuration.no_of_refreshes

            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods,self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]
            
            # Add random noise to the distribution if set True
            if(noise):
                self.occupancy = randomnoiser(self.occupancy, configuration.sample_size, configuration.noise_percentage)

            print('Linear distribution created.')
        else:
            raise ValueConfigurationError()

    # Get current occupnacy of the carpark
    def get_occupancy_level(self, step = None):
        return self.occupancy[0]

# LinearDistribution class
# Generates a linear distribution where gradient > 0 or gradient < = (i.e. y=mx+c)
# Config Value: gradient-line
class LinearDistribution(Distribution):
    def __init__(self, configuration, noise=False):
        if(isinstance(configuration,CarParkConfiguration)):
            print('Starting to create a linear distribution.')
            self.time_step = range(0, configuration.no_of_refreshes)
            gradient = (configuration.sample_size - configuration.min_occupancy)/configuration.no_of_refreshes
            occupancy = np.arange(configuration.min_occupancy, configuration.sample_size, gradient)

            self.occupancy = list(map(lambda z : round(z), occupancy))

            # Discretizing the distribtution into time steps
            if(len(configuration.selected_periods) > 1 or (not(configuration.selected_periods[0][0] == 1 and configuration.selected_periods[0][1] == configuration.no_of_refreshes))):
                res = trim_distribution(configuration.selected_periods,self.occupancy)
                self.time_step = res[0]
                self.occupancy = res[1]
            
            # Add random noise to the distribution if set True
            if(noise):
                self.occupancy = randomnoiser(self.occupancy, configuration.sample_size, configuration.noise_percentage)

            print('Linear distribution created.')
        else:
            raise ValueConfigurationError()

    # Get current occupnacy of the carpark
    def get_occupancy_level(self, step):
        closest_step = min(self.time_step, key=lambda x:abs(x-step))
        idx = self.time_step.index(closest_step)

        # Mapping continuous time into discrete timesteps 
        if(closest_step>step):
            return self.occupancy[idx-1]
        else:
            return self.occupancy[idx]

# Static Method
# Takes 2 distributions and merges them into a single distribution.
# paramters: dist1, dist2 = series of numbers
def super_impose(dist1, dist2):
    dists = []
    dists.extend(list(map(lambda x: (x, dist1.occupancy[dist1.time_step.index(x)]), dist1.time_step)))
    dists.extend(list(map(lambda x: (x, dist2.occupancy[dist2.time_step.index(x)]), dist2.time_step)))

    dists = sorted(dists, key=lambda x: x[0])

    d = {}
    for k,v in dists:
        if(k in d):
            d[k] = (d[k][0]+1, ((d[k][0]*d[k][1])+v)/(d[k][0]+1))
        else:
            d[k] = (1,v)

    return list(map(lambda x : x[0], d.items())), list(map(lambda x : round(x[1][1]), d.items()))

# Static Method
# Builds the normal distribution from the random data collection
# normalizes that with in the given parameters (fitting).
# parameters: init_dist = random data collection, sample_size = maximum value of representive slots
def normalizing_distribution(init_dist, sample_size):
    constant = sample_size/sum(init_dist)
    dist = map(lambda x: x * constant, init_dist)

    time_step = 1
    cum_occupancy = 0
    last_occupancy = 0

    time_step_list = [0]
    occupancy_list = [last_occupancy]

    for data in list(dist):
        cum_occupancy = cum_occupancy + data
        if(abs(cum_occupancy - last_occupancy) >= 1):
            last_occupancy = trunc(cum_occupancy)
            time_step_list.append(time_step)
            occupancy_list.append(last_occupancy)
        time_step = time_step + 1

    return (time_step_list, occupancy_list)

# Static Method
# Trims the distributions into selected discrete planning periods 
# parameters: selections = indexes of the selected planning periods, dist = full distribution 
def trim_distribution(selections, dist):
    occupancy = []
    for sel in selections:
        occupancy.extend(dist[sel[0]:sel[1]+1])
    time_step = range(1,len(occupancy)+1)
    return (occupancy,time_step)

# Static Method
# Smooths the curve using Gaussian 1-D filtering
def smooth(x,std):
    y = gaussian_filter1d(x, std)
    return list(map(lambda z : round(z), y))

# Static Method
# Adds random noise to the distribution
# parameters: x = distribution, up_bound = max value of the noise, coverage = percentage of noise in the distbution
def randomnoiser(x, up_bound, coverage):
    if(coverage <= 1.0 and coverage > 0.0):  
        noise = np.random.normal(0,2,len(x))   
        indexes = random.sample(range(len(x)), round(len(x)*coverage))
        for i in indexes:
            noise[i] = 0
        return list(map(lambda z : check_value(round(x[z] + noise[z]), up_bound), range(0,len(x)))) 
    else:
        return x

# Static Method
def check_value(x, up_bound):
    if(x<0):
        return 0
    elif(x>up_bound):
        return up_bound
    else:
        return x