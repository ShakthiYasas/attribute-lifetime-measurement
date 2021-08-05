from constants import strategy
from reactive import Reactive
from adaptive import Adaptive
from greedy import Greedy

class StrategyFactory:
    selected_algo = None

    def __init__(self, strat_name):
        if(strat_name in strategy):
            if('reactive'):
                self.selected_algo = Reactive()
            if('adaptive'):
                self.selected_algo =  Adaptive()
            if('greedy'):
                self.selected_algo =  Greedy()
        else:
            self.selected_algo =  Reactive()
