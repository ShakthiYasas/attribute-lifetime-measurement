import threading
import numpy as np
from datetime import datetime
from agents.agent import Agent
from agents.exploreagent import RandomAgent, MRUAgent, MFUAgent
from tensorflow.python.framework.ops import disable_eager_execution

from keras.models import Model 
from keras.optimizers import Adam
from keras.layers import Input, Dense

from agents.classifier.linkedcluster import LinkedCluster

disable_eager_execution()

class ACAgent(Agent):
    __hashLock = threading.Lock()

    def __init__(self, config, caller):
        # General Configurations
        self.__value_size = 1
        self.__caller = caller
        self.__reward_threshold = config.reward_threshold

        # Extrapolation ranges 
        self.__short = config.short
        self.__mid = config.mid
        self.__long = config.long

        # Setting the hyper-parameters
        self.__actor_lr = config.learning_rate 
        self.__critic_lr = config.learning_rate*5

        self.__gamma = config.discounting_factor
        self.__epsilons_max = config.e_greedy_max
        self.__epsilons_increment = config.e_greedy_increment
        self.__epsilons_decrement = config.e_greedy_decrement

        # Overriding the action space as cache or not cache
        self.action_space = [0,1]

        # Setting the features
        self.__n_features = 15

        # e-Greedy Exploration  
        self.__epsilons = list(config.e_greedy_init)
        self.__dynamic_e_greedy_iter = config.dynamic_e_greedy_iter
        if ((config.e_greedy_init is None) or (config.e_greedy_decrement is None)):
            self.__epsilons = list(self.epsilons_min)
        
        # Selecting the algorithem to execute during the exploration phase
        self.__explore_mentor = None
        if (config.explore_mentor.lower() == 'mru'):
            self.__explore_mentor = MRUAgent()
        elif (config.explore_mentor.lower() == 'mfu'):
            self.__explore_mentor = MFUAgent()
        else:
            self.__explore_mentor = RandomAgent()
               
        # Accumalated count of learning epochs 
        self.__learn_step_counter = 0

        # History of rewards
        self.reward_history = []

        # Initializing state space clustering algorithm
        self.__stateclusters = LinkedCluster(config.cluster_similarity_threshold, 
                    config.subcluster_similarity_threshold, config.pair_similarity_maximum)

        # Setting up Actor and Critic Networks
        self.__critic = self.__build_critic()
        self.__actor, self.__policy = self.__build_actor()

        # Cluster, Entity, Attribute Mapping for currently executing operations
        self.__cluster_context_map = {}
        
    # Approximates the policy and value using the Neural Network
    # Actor: state is input and probability of each action is output of model
    def __build_actor(self):
        delta = Input(shape=[1])

        # NN Layers
        input_layer = Input(shape=(self.__n_features,))
        hidden_layer = Dense(8, activation='relu', kernel_initializer='he_uniform')(input_layer)
        output_layer = Dense(len(self.action_space), activation='softmax')(hidden_layer)

        actor = Model(inputs=[input_layer,delta], outputs=[output_layer])
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.__actor_lr))

        policy = Model(inputs=[input_layer], outputs=[output_layer])
       
        return actor, policy

    # Critic: state is input and value of state is output of model
    def __build_critic(self):
        # NN Layers
        input_layer = Input(shape=(self.__n_features,))
        hidden_layer = Dense(8, activation='relu', kernel_initializer='he_uniform')(input_layer)
        output_layer = Dense(self.__value_size, activation='softmax')(hidden_layer)

        critic = Model(inputs=[input_layer], outputs=[output_layer])
        critic.compile(loss='mean_squared_error', optimizer=Adam(lr=self.__critic_lr))
        
        return critic

    # Map this observation to the state space.
    # This is done by predicting with cluster that that this observation falls into.
    def modify_state_space(self, observation, is_cached=False):
        idx = self.__stateclusters.predict(np.array(observation['features'])[np.newaxis, :])
        key = (observation['entityid'], observation['attribute'])
        if(key in self.__cluster_context_map):
            if(not is_cached):
                self.__hashLock.acquire()
                value = self.__cluster_context_map[key]
                self.__cluster_context_map[key] = (value[0], value[1]+1, datetime.now())
                self.__hashLock.release()

            return self.__cluster_context_map[key][0]
        else:
            if(not is_cached):
                self.__cluster_context_map[key] = (idx, 1, datetime.now())

            return idx

    # Select the most suitable action given a state
    # The biggest challenge here is the change of state with changes in the environment. 
    # So, there is no gurantee that the state will remain unchanged. 
    # So, the only option is to find the state that is the most similar using clustering.
    def choose_action(self, observation, skip_random=False): 
        obs = observation['features']
        entityid = observation['entityid']
        attribute = observation['attribute']

        probabilities = self.__policy.predict(obs)[0]
        action = np.random.choice(len(self.action_space), 1, p=probabilities)

        if(action == 0):
            # Item is defenitly not to be cached
            random_value = np.random.uniform()
            if(random_value < self.__epsilons and not skip_random):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    action = self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
                    if(action != (0,0)):
                        cached_lifetime = 10 # This need to set
                        return (action, (cached_lifetime, 0))
                    else:
                        delay_time = 10 # This need to be set
                        return (action, (0, delay_time))
                else:
                    action = self.__explore_mentor.choose_action(self.__caller.get_observed())
                    if(action != (0,0)):
                        cached_lifetime = 10 # This need to set
                        return [(action, (cached_lifetime, 0))]
                    else:
                        delay_time = 10 # This need to be set
                        return (action, (0, delay_time))
            else:
                delay_time = 10 # This need to be set
                return (action, (0, delay_time))
        else:
            # Decided to be cached
            translated_action = (entityid, attribute) 
            cached_lifetime = 10 # This need to set
            return (translated_action, (cached_lifetime, 0))

    # Learn the network
    def learn(self, state, action, reward, next_state):
        critic_value = self.__critic.predict(state)
        new_critic_value = self.__critic.predict(next_state)

        target_value = reward + self.__gamma*new_critic_value
        delta = target_value - critic_value

        actions = np.zeros([1, len(self.action_space)])
        actions[np.arange(1), action] = 1.0

        self.__actor.fit([state, delta], actions, verbose=0)
        self.__critic.fit(state, target_value, verbose=0)

        # Increasing or Decreasing epsilons
        if self.__learn_step_counter % self.__dynamic_e_greedy_iter == 0:
            # If e-greedy
            if self.__epsilons_decrement is not None:
                # Dynamic bidirectional e-greedy 
                # That allows the netowrk to be self adaptive between exploration and exploitation depending on the
                # current performance of the system. i.e., if MR is high, the increase epsilon (more exploration)
                # and on the contrary, decrease epsilon if MR is less (more exploitation). 
                if self.__epsilons_increment is not None:
                    rho = np.mean(np.array(self.reward_history))
                    if rho >= self.__reward_threshold:
                        self.__epsilons -= self.__epsilons_decrement
                    else:
                        self.__epsilons += self.__epsilons_increment              
                # Traditional e-greedy
                else:
                    self.__epsilons -= self.__epsilons_decrement

            # Enforce upper bound and lower bound
            truncate = lambda x, lower, upper: min(max(x, lower), upper)
            self.__epsilons = truncate(self.__epsilons, self.epsilons_min, self.__epsilons_max)

        self.__learn_step_counter = 0 if self.__learn_step_counter > 1000 else self.__learn_step_counter + 1
