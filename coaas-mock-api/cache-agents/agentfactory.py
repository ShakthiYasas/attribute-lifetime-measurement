from agent import Agent
from dqnagent import DQNAgent
from a3cagent import A3CAgent

from configurations.cacheagentconfig import DQNConfiguration, A3CConfiguration

# Instantiate a singleton agent instance according to configuration
class AgentFactory:
    # Class varaible
    agent=None
    configuration = None

    def __init__(self, configuration = None):
        if(configuration != None):
            self.configuration = configuration

    # Retruns the singleton instance of a cache
    def get_agent(self, actions, features) -> Agent:
        if(self.agent == None):
            if(self.configuration != None and isinstance(self.configuration,DQNConfiguration)):
                return DQNAgent(actions, features, self.configuration)
            if(self.configuration != None and isinstance(self.configuration,A3CConfiguration)):
                return A3CAgent(actions, features, self.configuration)
            else:
                raise ValueError('Invalid configuration.')
        else:
            return self.agent