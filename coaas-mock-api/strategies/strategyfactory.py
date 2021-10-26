from constants import strategy

from strategies.reactive import Reactive
from strategies.adaptive import Adaptive
from strategies.greedy import Greedy
from strategies.greedy import Strategy

class StrategyFactory:
    __selected_algo = None

    def __init__(self, strat_name, db, window, isstatic=True):
        if(strat_name in strategy):
            if(strat_name == 'reactive'):
                self.__selected_algo = Reactive(db, window, isstatic)
            if(strat_name == 'adaptive'):
                self.__selected_algo =  Adaptive(db, window, isstatic)
            if(strat_name == 'greedy'):
                self.__selected_algo =  Greedy(db, window, isstatic)
        else:
            self.__selected_algo =  Reactive(db, window, isstatic)

    # Retruns the singleton instance of a strategy
    def get_cache_memory(self) -> Strategy:
        return self.__selected_algo