from constants import strategy
from strategies.reactive import Reactive
from strategies.adaptive import Adaptive
from strategies.greedy import Greedy

class StrategyFactory:
    selected_algo = None

    def __init__(self, strat_name, attributes, url, db):
        if(strat_name in strategy):
            if(strat_name == 'reactive'):
                self.selected_algo = Reactive(attributes, url, db)
            if(strat_name == 'adaptive'):
                self.selected_algo =  Adaptive(attributes, url, db)
            if(strat_name == 'greedy'):
                self.selected_algo =  Greedy(attributes, url, db)
        else:
            self.selected_algo =  Reactive(attributes, url, db)
