import math
import constants as const

class Configuration:
    """Base class for Configuartions"""
    pass

# Configuration of the Car Park
class CarParkConfiguration(Configuration):
    sample_size = 100 # total available (constrait).
    total_time = 600000 # In miliseconds. 10 minutes.
    skew = 0 # median skewness 
    standard_deviation = 2 # Using the emperical rule of 95% representation
    sampling_rate = 1000 # sampling every 1000ms = 1s
    no_of_refreshes = 0 # number of times the car park has refreshed data during the total time period
    variation=1 #-1 for decreasing, 0 for random, +1 for increasing
    median=0

    def __init__(self, sample_size = None, standard_deviation = None, total_time = None, skew = None, sampling_rate = None, variation = None):
        
        if(sample_size != None and sample_size > 0):
            self.sample_size = sample_size
        if(standard_deviation != None and standard_deviation > 0):
            self.standard_deviation = standard_deviation
        if(total_time != None and total_time > 60000):
            self.total_time = total_time
        if(skew != None):
            self.skew = skew
        if(sampling_rate != None and sampling_rate > 0 and sampling_rate <= self.total_time):
            self.sampling_rate = sampling_rate
        if(variation != None):
            self.variation = const.variations[variation.lower()]
        
        self.no_of_refreshes =  int(self.total_time/self.sampling_rate)
        self.median = self.no_of_refreshes/2

        print('Configuration:\n\tSample Size = '+str(self.sample_size)+'\n\tStandard Distruntion = '+str(self.standard_deviation)
        +'\n\tTotal Time = '+str(int(math.ceil(self.total_time/60000)))+const.mins
        +'\n\tSkew = '+str(self.skew)
        +'\n\tSampling Interval = '+str(self.sampling_rate)
        +'\n\tNo of Refreshes = '+str(self.no_of_refreshes)
        +'\n\tTrend = '+str(self.variation))

