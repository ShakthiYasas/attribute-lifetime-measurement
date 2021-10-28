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
                print('Initializing a DQN based RL agent for selective context caching.')
                self.__agent = DQNAgent(actions, features, self.__configuration)
            if(self.__configuration != None and isinstance(self.__configuration,A3CConfiguration)):
                print('Initializing a Asynchronous-Advantage-Actor-Critic based RL agent for selective context caching.')
                self.__agent = A3CAgent(actions, features, self.__configuration)
            else:
                raise ValueError('Invalid configuration.')
        
        return self.__agent