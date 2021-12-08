import time
import queue
import threading
import statistics
from datetime import datetime

from agents.agent import Agent
from lib.fifoqueue import FIFOQueue_2
from lib.event import post_event_with_params
from agents.exploreagent import RandomAgent, MRUAgent, MFUAgent

import numpy as np
import keras.backend as kb
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from tensorflow.python.framework.ops import disable_eager_execution

from agents.classifier.linkedcluster import LinkedCluster

MAX_WORKERS = 50

LAYER_1_NEURONS = 512
LAYER_2_NEURONS = 256

ACTOR_MODEL_PATH = 'agents/saved-models/ac/actor-model'
CRITIC_MODEL_PATH = 'agents/saved-models/ac/critic-model'
POLICY_MODEL_PATH = 'agents/saved-models/ac/policy-model'

class ACAgent(threading.Thread, Agent):
    __hashLock = threading.Lock()

    def __init__(self, config, caller):
        disable_eager_execution()
        
        # Thread
        self.q = queue.Queue()
        self.timeout = 1.0/60
        super(ACAgent, self).__init__()

        # General Configuration
        self.__value_size = 1
        self.__caller = caller
        self.__midtime =  (config.mid*config.window)/1000
        self.__longtime = (config.long*config.window)/1000
        self.__reward_threshold = config.reward_threshold

        # Setting the hyper-parameters
        self.__actor_lr = config.learning_rate # i.e. 0.0001
        self.__critic_lr = config.learning_rate*10 # i.e. 0.001

        self.__gamma = config.discount_rate
        self.__epsilons_max = config.e_greedy_max
        self.__epsilons_increment = config.e_greedy_increment
        self.__epsilons_decrement = config.e_greedy_decrement

        # Overriding the action space as cache or not cache
        self.action_space = [0,1]

        # Setting the features
        self.__n_features = 15

        # e-Greedy Exploration  
        self.__epsilons = config.e_greedy_init
        self.__dynamic_e_greedy_iter = config.dynamic_e_greedy_iter
        if ((config.e_greedy_init is None) or (config.e_greedy_decrement is None)):
            self.__epsilons = self.epsilons_min
        
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
        self.reward_history = FIFOQueue_2(100)

        # Initializing state space clustering algorithm
        self.__stateclusters = LinkedCluster(config.cluster_similarity_threshold, 
                    config.subcluster_similarity_threshold, config.pair_similarity_maximum)

        # Cluster, Entity, Attribute Mapping for currently executing operations
        self.__cluster_context_map = {}
    
    def onThread(self, function, *args, **kwargs):
        self.q.put((function, args, kwargs))
    
    def run(self):
        # Setting up Actor and Critic Networks
        self.actor, self.policy, self.critic = self.__build_actor_critic()
        while True:
            try:
                function, args, kwargs = self.q.get(timeout=self.timeout)
                _ = function(*args, **kwargs)

            except queue.Empty:
                self.__idle()

    def __idle(self):
        time.sleep(0.5) 

    # Approximates the policy and value using the Neural Network
    # Actor: state is input and probability of each action is output of model
    def __build_actor_critic(self):
        # NN Layers  
        input_layer = Input(shape=(self.__n_features,))
        hidden_layer = Dense(LAYER_1_NEURONS, activation='relu')(input_layer)
        hidden_layer_2 = Dense(LAYER_2_NEURONS, activation='relu')(hidden_layer)
        value_layer = Dense(self.__value_size, activation='linear')(hidden_layer_2)
        output_layer = Dense(len(self.action_space), activation='softmax')(hidden_layer_2)
        
        try:
            # Loading Existing Models
            # Useful for Fault recovery and re-starts
            actor = keras.models.load_model(ACTOR_MODEL_PATH)
            critic = keras.models.load_model(CRITIC_MODEL_PATH)
            policy = keras.models.load_model(POLICY_MODEL_PATH)

            return actor, policy, critic
        except(Exception):
            # Actor Model
            delta = Input(shape=[1])
            actor = Model(inputs=[input_layer,delta], outputs=[output_layer])
            actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.__actor_lr))
            actor.save(ACTOR_MODEL_PATH)

            # Critic Model      
            critic = Model(inputs=[input_layer], outputs=[value_layer])
            critic.compile(loss='mean_squared_error', optimizer=Adam(lr=self.__critic_lr))
            critic.save(CRITIC_MODEL_PATH)

            # Policy Model 
            policy = Model(inputs=[input_layer], outputs=[output_layer])
            policy.save(POLICY_MODEL_PATH)
        
            return actor, policy, critic

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
    def choose_action(self, paramters): 
        observation = paramters[0]
        skip_random = paramters[1]
        ref_key = None
        if(len(paramters)>2):
            ref_key = paramters[2]

        obs = np.asarray(observation['features'])[np.newaxis, :]
        entityid = observation['entityid']
        attribute = observation['attribute']

        probabilities = self.policy.predict(obs, verbose=0)[0]
        action = 1
        if(probabilities[0] > probabilities[1]):
            action = 0

        print(str(action)+' for entity:'+str(entityid)+' and att:'+str(attribute))
        print()

        if(action == 0):
            # Item is defenitly not to be cached
            delay_time = self.__calculate_delay(probabilities[action])
            post_event_with_params("subscribed_actions", (entityid, attribute, 0, delay_time, 0, observation['features'], ref_key))

            random_value = np.random.uniform()
            if(random_value <= self.__epsilons and not skip_random):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    action = self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
                    if(action != (0,0)):
                        observation = self.__caller.get_feature_vector(action[0], action[1])
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation, ref_key))
                else:
                    action = self.__explore_mentor.choose_action(self.__caller.get_observed())
                    if(action != (0,0)):
                        observation = self.__caller.get_feature_vector(action[0], action[1])
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation, ref_key)) 
        else:
            # Decided to be cached
            cached_lifetime = self.__calculate_expected_cached_lifetime(entityid, attribute, probabilities[action])
            post_event_with_params("subscribed_actions", (entityid, attribute, cached_lifetime, 0, 1, observation['features'], ref_key))

        return 0

    def __calculate_expected_cached_lifetime(self, entityid, attr, prob):
        cached_lt_res = self.__caller.get_access_to_db().read_all_with_limit('attribute-cached-lifetime',{
                    'entity': entityid,
                    'attribute': attr
                },10)
        if(cached_lt_res):
            avg_lt = statistics.mean(list(map(lambda x: x['c_lifetime'], cached_lt_res)))
            return (avg_lt/0.5)*(prob-0.5)
        else:
            return self.__midtime
    
    def __calculate_delay(self, prob):
        return (self.__longtime/0.5)*(prob-0.5)

    # Learn the network
    def learn(self, parameters):
        state = parameters[0]
        action = parameters[1]
        reward = parameters[2]
        next_state = parameters[3]

        self.reward_history.push(reward)

        state = np.array(state)[np.newaxis, :]
        next_state = np.array(next_state['features'])[np.newaxis, :]

        critic_value = self.critic.predict(state)
        new_critic_value = self.critic.predict(next_state)

        target_value = reward + self.__gamma*new_critic_value
        delta = target_value - critic_value

        actions = np.zeros([1, len(self.action_space)])
        actions[np.arange(1), action] = 1.0

        # Re-learning the model
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target_value, verbose=0)

        # Saving the model again as an update
        # print('This issue start to happen when learn is called?')
        self.actor.save(ACTOR_MODEL_PATH)
        self.critic.save(CRITIC_MODEL_PATH)

        # print('After Saving!')

        self.__learn_step_counter+=1
        # print('Leaning Step Counter: '+str(self.__learn_step_counter))

        # Increasing or Decreasing epsilons
        if self.__learn_step_counter % self.__dynamic_e_greedy_iter == 0:
            # If e-greedy
            if self.__epsilons_decrement is not None:
                # Dynamic bidirectional e-greedy 
                # That allows the netowrk to be self adaptive between exploration and exploitation depending on the
                # current performance of the system. i.e., if MR is high, the increase epsilon (more exploration)
                # and on the contrary, decrease epsilon if MR is less (more exploitation). 
                if self.__epsilons_increment is not None:
                    rho = np.mean(np.array(self.reward_history.getlist()))
                    if rho >= self.__reward_threshold:
                        self.__epsilons -= self.__epsilons_decrement
                    else:
                        self.__epsilons += self.__epsilons_increment              
                # Traditional e-greedy
                else:
                    self.__epsilons -= self.__epsilons_decrement

            # Enforce upper bound and lower bound
            if(self.__epsilons < self.epsilons_min):
                self.__epsilons = self.epsilons_min
            elif(self.__epsilons > self.__epsilons_max):
                self.__epsilons = self.__epsilons_max
        
        self.__learn_step_counter = 0 if self.__learn_step_counter > 1000 else self.__learn_step_counter + 1
    
    # Get the value of the epsilon value now
    def get_current_epsilon(self):
        return self.__epsilons
    
    # Get current discunt rate
    def get_discount_rate(self):
        return self.__gamma