import math
import constants as const

class Configuration:
    """Base class for Configuartions"""
    pass

# Configuration of the Car Park
class CarParkConfiguration(Configuration):
    sample_size = 100 # total available (constrait).
    total_time = 600000 # In miliseconds. 10 minutes.
    skew = [0] # median skewness 
    standard_deviation = [2] # Using the emperical rule of 95% representation
    sampling_rate = 1000 # sampling every 1000ms = 1s
    no_of_refreshes = 0 # number of times the car park has refreshed data during the total time period
    variation=[] #-1 for decreasing, 0 for random, +1 for increasing
    median=0
    planning_period = 60000 # Default 1 minute period
    selected_periods = []

    def __init__(self, sample_size = None, standard_deviation = None, total_time = None, skew = None, sampling_rate = None, variation = None,
        planning_period = None, selected_periods = None):
        
        if(sample_size != None and sample_size > 0):
            self.sample_size = sample_size
        if(standard_deviation != None):
            tk = standard_deviation.split(',')
            self.standard_deviation = list(map(lambda x: float(x) if float(x) > 0 else 2.0, tk ))
        if(total_time != None and total_time > 60000):
            self.total_time = total_time
        if(skew != None):
            self.skew = list(map(lambda x : float(x), skew.split(',')))
        if(sampling_rate != None and sampling_rate > 0 and sampling_rate <= self.total_time):
            self.sampling_rate = int(sampling_rate)
        if(variation != None):
            for var in variation:
                self.variation.append(const.variations[var.lower()])
        else:
            self.variation.append(1)
        if(planning_period != None and int(planning_period) > self.sampling_rate):
            self.planningPeriod = int(planning_period)

        self.no_of_refreshes =  int(self.total_time/self.sampling_rate)
        self.median = self.no_of_refreshes/2

        if(selected_periods != None):
            if(selected_periods == '*'):
                self.selected_periods.append((1,self.no_of_refreshes))
            else:
                no_of_sections = self.total_time/self.planningPeriod
                selected_periods = list(map(lambda x: int(x), selected_periods.split(',')))
                if(all(i <= no_of_sections for i in selected_periods)):
                    for selected in selected_periods:
                        low_bound = (planning_period*(selected-1))/self.sampling_rate
                        up_bound = planning_period*selected/self.sampling_rate
                        self.selected_periods.append((round(low_bound),round(up_bound)))
        else:
            self.selected_periods.append((1,self.no_of_refreshes))

        

        print('Configuration:\n\tSample Size = '+str(self.sample_size)+'\n\tStandard Distruntion = '+str(self.standard_deviation)
        +'\n\tTotal Time = '+str(int(math.ceil(self.total_time/60000)))+const.mins
        +'\n\tSkew = '+str(self.skew)
        +'\n\tSampling Interval = '+str(self.sampling_rate)
        +'\n\tNo of Refreshes = '+str(self.no_of_refreshes)
        +'\n\tTrend = '+str(self.variation))

