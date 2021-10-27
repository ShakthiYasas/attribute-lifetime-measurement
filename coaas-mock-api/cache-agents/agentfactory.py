from agent import Agent
from dqnagent import DQNAgent
from a3cagent import A3CAgent

from configurations.agentconfig import DQNConfiguration, A3CConfiguration

# Instantiate a singleton agent instance according to configuration
class AgentFactory:
    # Class varaible
    __agent=None
    __configuration = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.__configuration = configuration

    # Retruns the singleton instance of a cache
    def get_agent(self, actions, features) -> Agent:
        if(self.__agent == None):
            if(self.__configuration != None and isinstance(self.__configuration,DQNConfiguration)):
                return DQNAgent(actions, features, self.__configuration)
            if(self.__configuration != None and isinstance(self.__configuration,A3CConfiguration)):
                return A3CAgent(actions, features, self.__configuration)
            else:
                raise ValueError('Invalid configuration.')
        else:
            return self.__agent