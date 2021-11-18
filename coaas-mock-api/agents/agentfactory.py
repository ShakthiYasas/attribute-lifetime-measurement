from agents.agent import Agent
from agents.acagent import ACAgent
from agents.dqnagent import DQNAgent
from agents.a3cagent import A3CAgent
from agents.simpleagent import SimpleAgent
from agents.ddpgacagent import DDPGACAgent

# from configurations.agentconfig import DQNConfiguration, A3CConfiguration,
from configurations.agentconfig import ACConfiguration, SimpleConfiguration, DDPGConfiguration

# Instantiate a singleton agent instance according to configuration
class AgentFactory:
    # Class varaible
    __agent=None
    __configuration=None

    def __init__(self, type, configuration = None, caller = None):
        self.__caller = caller
        if(configuration != None):
            
            if(type == 'simple'):
                self.__configuration = SimpleConfiguration(configuration)
            elif(type == 'ac'):
                self.__configuration = ACConfiguration(configuration)
            elif(type == 'ddpg'):
                self.__configuration = DDPGConfiguration(configuration)
            # elif(type == 'a3c'):
            #     self.__configuration = A3CConfiguration(configuration)
            # elif(type == 'dqn'):
            #    self.__configuration = DQNConfiguration(configuration)
            else:
                raise ValueError('Invalid RL agent type.')

    # Retruns the singleton instance of an RL agent
    def get_agent(self) -> Agent:
        if(self.__agent == None):
            if(self.__configuration != None and isinstance(self.__configuration,SimpleConfiguration)):
                print('Initializing a Simple NPV based selective context caching.')
                self.__agent = SimpleAgent(self.__configuration, self.__caller)
            elif(self.__configuration != None and isinstance(self.__configuration,ACConfiguration)):
                print('Initializing a Actor-Critic based RL agent for selective context caching.')
                self.__agent = ACAgent(self.__configuration, self.__caller)
            elif(self.__configuration != None and isinstance(self.__configuration,DDPGConfiguration)):
                print('Initializing a Deep-Detterministic-Policy-Gradient-Actor-Critic based RL agent for selective context caching.')
                self.__agent = DDPGACAgent(self.__configuration, self.__caller)
            # elif(self.__configuration != None and isinstance(self.__configuration,DQNConfiguration)):
            #    print('Initializing a DQN based RL agent for selective context caching.')
            #    self.__agent = DQNAgent(self.__configuration, self.__caller)
            # elif(self.__configuration != None and isinstance(self.__configuration,A3CConfiguration)):
            #    print('Initializing a Asynchronous-Advantage-Actor-Critic based RL agent for selective context caching.')
            #    self.__agent = A3CAgent(self.__configuration, self.__caller)
            else:
                raise ValueError('Invalid configuration.')
        
        return self.__agent