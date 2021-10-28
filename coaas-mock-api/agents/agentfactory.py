from agent import Agent
from dqnagent import DQNAgent
from a3cagent import A3CAgent

from configurations.agentconfig import DQNConfiguration, A3CConfiguration

# Instantiate a singleton agent instance according to configuration
class AgentFactory:
    # Class varaible
    __agent=None
    __configuration = None

    def __init__(self, type, configuration = None):
        if(configuration != None):
            if(type == 'a3c'):
                self.__configuration = A3CConfiguration(configuration)
            elif(type == 'dqn'):
                self.__configuration = DQNConfiguration(configuration)
            else:
                raise ValueError('Invalid RL agent type.')

    # Retruns the singleton instance of an RL agent
    def get_agent(self) -> Agent:
        if(self.__agent == None):
            if(self.__configuration != None and isinstance(self.__configuration,DQNConfiguration)):
                print('Initializing a DQN based RL agent for selective context caching.')
                self.__agent = DQNAgent(self.__configuration)
            if(self.__configuration != None and isinstance(self.__configuration,A3CConfiguration)):
                print('Initializing a Asynchronous-Advantage-Actor-Critic based RL agent for selective context caching.')
                self.__agent = A3CAgent(self.__configuration)
            else:
                raise ValueError('Invalid configuration.')
        
        return self.__agent