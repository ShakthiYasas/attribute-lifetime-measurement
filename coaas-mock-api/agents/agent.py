# Abstract class
class Agent(object):
    action_space = [0,1]
    def __init__(self): pass
    def choose_action(self, observation): pass
    def store_transition(self, state, action, reward, new_state): pass

class ExplorationAgent(Agent):
    @staticmethod
    def _choose_action(): pass
    @staticmethod
    def _choose_action(observation): pass