from constants import strategy
from reactive import Reactive
from adaptive import Adaptive
from greedy import Greedy

class StrategyFactory:
    selected_algo = None

    def __init__(self, strat_name, attributes, db):
        if(strat_name in strategy):
            if('reactive'):
                self.selected_algo = Reactive(attributes, db)
            if('adaptive'):
                self.selected_algo =  Adaptive(attributes, db)
            if('greedy'):
                self.selected_algo =  Greedy(attributes, db)
        else:
            self.selected_algo =  Reactive(attributes, db)
