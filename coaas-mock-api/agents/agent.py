# Abstract class
class Agent(object):
    MIN_CACHE_LIFE = 4
    action_space = [(0,0)]
    epsilons_min = 0.001
    def __init__(self, config, caller): pass
    def choose_action(self, observation, skip_random=False): pass
    def store_transition(self, state, action, reward, new_state): pass

class ExplorationAgent():
    @staticmethod
    def _choose_action(): pass
    def choose_action(self, observation:dict): pass