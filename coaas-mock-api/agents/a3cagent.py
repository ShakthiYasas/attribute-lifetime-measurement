from agents.agent import Agent

class A3CAgent(Agent):
    ##############################################################################################
    # Asynchronous Advantage Actor Critc is one of the future directions for this problem. 
    # That is to manage large number of concurrent requests.
    ##############################################################################################

    def __init__(self, n_actions, caller): pass
    def choose_action(self, observation, skipRandom=False): pass
    def store_transition(self, s, a, r, s_): pass