# Abstract class
class Agent(object):
    def __init__(self, n_actions): pass
    def choose_action(self, observation): pass
    def store_transition(self, s, a, r, s_): pass