import numpy as np
from agent import Agent
from exploreagent import RandomAgent, MRUAgent, MFUAgent

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class ACAgent(Agent):
    def __init__(self, config, caller):
        # General Configurations
        self.__value_size = 1
        self.__caller = caller
        self.__reward_threshold = config.reward_threshold

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
        self.__feature_vector = [
            'short_term_access (0)', 'expected_short_term_access (1)', 
            'mid_term_access (2)', 'expected_mid_term_access (3)',
            'long_term_access (4)', 'expected_long_term_access (5)',   
            'short_term_hitrate (6)', 'expected_short_term_hitrate (7)',
            'mid_term_hitrate (8)', 'expected_mid_term_hitrate (9)',
            'long_term_hitrate (10)', 'expected_long_term_hitrate (11)',         
            'average_cached_lifetime (12)', 
            'average_latency (13)',
            'average_retrieval_cost (14)'
            ]
        self.__n_features = len(self.__feature_vector)

        # e-Greedy Exploration
        self.__dynamic_e_greedy_iter = config.dynamic_e_greedy_iter
        self.__epsilons = list(config.e_greedy_init)
        if (config.e_greedy_init is None) or (config.e_greedy_decrement is None):
            self.__epsilons = list(self.epsilons_min)
        
        # Selecting the algorithem to execute during the exploration phase
        self.__explore_mentor = None
        if config.explore_mentor.lower() == 'mru':
            self.__explore_mentor = MRUAgent()
        elif config.explore_mentor.lower() == 'mfu':
            self.__explore_mentor = MFUAgent()
        else:
            self.__explore_mentor = RandomAgent()
               
        # Accumalated count of learning epochs 
        self.__learn_step_counter = 0

        # History of rewards
        self.reward_history = []

        # Setting up Actor and Critic Networks
        self.__actor = self.__build_actor()
        self.__critic = self.__build_critic()

    # Approximates the policy and value using the Neural Network
    # Actor: state is input and probability of each action is output of model
    def __build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.__n_features, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(len(self.action_space), activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.__actor_lr))
       
        return actor

    # Critic: state is input and value of state is output of model
    def __build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.__n_features, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.__value_size, activation='softmax',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.__critic_lr))
        
        return critic

    # Select the most suitable action given a state
    # The biggest challenge here is the change of state with changes in the environment. 
    # So, there is no gurantee that the state will remain unchanged. 
    # So, the only option is to find the state that is the most similar using clustering.
    def choose_action(self, observation): 
        random_value = np.random.uniform()
        if(random_value < self.__epsilons):
            if(isinstance(self.__explore_mentor,MFUAgent)):
                return self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
            else:
                return self.__explore_mentor.choose_action(self.__caller.get_observed())
                # Here the action is a (entityid,attribute) to cache
        else:
            # This needs to be looked into
            policy = self.actor.predict(np.array(observation['features']), batch_size=1).flatten()
            base_action = np.random.choice(len(self.action_space), 1, p=policy)[0]
            return (observation['entityid'], observation['attribute']) if base_action > 0.7 else (0,0)

    # Update the transition probabilities
    def store_transition(self, s, a, r, s_): 
        pass

    # Learn the network
    def learn(self, state, action, reward, next_state):
        target = np.zeros((1, self.__value_size))
        advantages = np.zeros((1, len(self.action_space)))

        value = self.__critic.predict(state)[0]
        next_value = self.__critic.predict(next_state)[0]

        advantages[0][action] = reward + self.__gamma * (next_value) - value
        target[0][0] = reward + self.__gamma * next_value

        self.__actor.fit(state, advantages, epochs=1, verbose=0)
        self.__critic.fit(state, target, epochs=1, verbose=0)

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

    # Getter method for feature vetcor labels
    def get_feature_vector(self):
        return self.__feature_vector