import time
import queue
import threading

import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform

from keras import regularizers
from agents.agent import Agent
from lib.fifoqueue import FIFOQueue_2
from lib.event import post_event_with_params
from agents.ddpg_helpers.actionnoise import ActionNoise
from agents.ddpg_helpers.replaybuffer import ReplayBuffer
from agents.exploreagent import RandomAgent, MRUAgent, MFUAgent

from tensorflow.python.framework.ops import disable_eager_execution

LAYER_1_NEURONS = 512
LAYER_2_NEURONS = 256

CHECKPOINT_PATH = 'agents/saved-models/ddpg/'
ACTOR_MODEL_PATH = 'agents/saved-models/ddpg/actor-model'
CRITIC_MODEL_PATH = 'agents/saved-models/ddpg/critic-model'

N_ACTIONS = 1 # Becuase, it's continous
N_FEATURES = 15

class DDPGACAgent(threading.Thread, Agent):
    def __init__(self, config, caller):
        disable_eager_execution()

        # Thread
        self.__timeout = 1.0/60
        self.__action_queue = queue.Queue()
        super(DDPGACAgent, self).__init__()

        # General Configuration
        self.__caller = caller
        self.__window = config.window
        self.__batch_size = config.batch_size
        self.__action_bound = config.max_action_value
        self.__midtime = (config.mid*config.window)/1000
        self.__reward_threshold = config.reward_threshold

        # Setting the hyper-parameters
        self.__actor_lr = config.learning_rate # i.e. 0.0001
        self.__critic_lr = config.learning_rate*10 # i.e. 0.001

        self.__tau = config.tau
        self.__gamma = config.discount_rate
        self.__epsilons_max = config.e_greedy_max
        self.__epsilons_increment = config.e_greedy_increment
        self.__epsilons_decrement = config.e_greedy_decrement

        # Initializing Replay Memory 
        self.__buffer = ReplayBuffer(config.history_size, self.__batch_size, N_FEATURES, N_ACTIONS)

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

    # Push a request to the queue to run a function of this class
    def onThread(self, function, *args, **kwargs):
        self.__action_queue.put((function, args, kwargs))
    
    # Executing the thread
    def run(self):
        # Initializing Tensorflow Session
        self.__session = tf.compat.v1.Session()

        # Actor-Critic Networks
        self.__critic = Critic(self.__critic_lr, 'Critic', self.__session)
        self.__actor = Actor(self.__actor_lr, 'Actor', self.__session, self.__action_bound, self.__batch_size)
        
        # Target Actor-Critic Networks
        self.__target_critic = Critic(self.__critic_lr, 'TargetCritic', self.__session)
        self.__target_actor = Actor(self.__actor_lr, 'TargetActor', self.__session, self.__action_bound, self.__batch_size)
        
        # Noise generator for the action space
        self.__noise = ActionNoise(np.zeros(N_ACTIONS))

        self.__update_critic = []
        for i in range(len(self.__target_critic.params)):
            self.__update_critic.append(self.__target_critic.params[i].assign(
                    tf.multiply(self.__critic.params[i], self.__tau) + tf.multiply(self.__target_critic.params[i], 1 - self.__tau)))
        
        self.__update_actor= []
        for i in range(len(self.__target_actor.params)):
            self.__update_actor.append(self.__target_actor.params[i].assign(
                    tf.multiply(self.__actor.params[i], self.__tau) + tf.multiply(self.__target_actor.params[i], 1 - self.__tau)))

        self.__session.run(tf.compat.v1.global_variables_initializer())
        self.__update_network_params(True)

        while True:
            try:
                function, args, kwargs = self.__action_queue.get(timeout=self.__timeout)
                function(*args, **kwargs)
            except queue.Empty:
                self.__idle()

    def __idle(self):
        time.sleep(0.5) 

    # Updating the paramters (weighs) of the actor and critic networks
    def __update_network_params(self, is_init=False):
        if(is_init):
            tau_copy = self.__tau
            self.__tau = 1.0
            self.__update_networks()
            self.__tau = tau_copy
        else:
            self.__update_networks()

    # Internal method for above
    def __update_networks(self):
        self.__target_critic.get_session().run(self.__update_critic)
        self.__target_actor.get_session().run(self.__update_actor)

    # Get the value of the epsilon value now
    def get_current_epsilon(self):
        return self.__epsilons

    # Select the most suitable action given a state
    def choose_action(self, paramters):
        observation = paramters[0]
        skip_random = paramters[1]

        obs = np.asarray(observation['features'])[np.newaxis, :]
        entityid = observation['entityid']
        attribute = observation['attribute']
        
        mu = self.__actor.predict(obs)
        noise = self.__noise()
        mu_prime = mu + noise
        in_time = (mu_prime[0] * self.__window)/1000

        if(mu_prime>0):
            # Estimated a positive cached lifetime. So, is decided to be cached
            post_event_with_params("subscribed_actions", (entityid, attribute, in_time[0], 0, 1, observation['features']))
        else:
            # Estimated a negative cached lifetime.
            # So, is not decided to be cached and the estimation is used to delay the next decision epoch.
            post_event_with_params("subscribed_actions", (entityid, attribute, 0, -mu_prime[0][0], 0, observation['features']))

            random_value = np.random.uniform()
            if(random_value < self.__epsilons and not skip_random):
                if(isinstance(self.__explore_mentor,MFUAgent)):
                    action = self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
                    if(action != (0,0)):
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation['features']))
                else:
                    action = self.__explore_mentor.choose_action(self.__caller.get_observed())
                    if(action != (0,0)):
                        post_event_with_params("subscribed_actions", (action[0], action[1], self.__midtime, 0, 1, observation['features']))
                
        
    # Modify and update the actor and critic networks
    # based on experience
    def learn(self, parameters):
        state = parameters[0]
        action = parameters[1]
        reward = parameters[2]
        new_state = parameters[3]

        self.reward_history.push(reward)

        state = np.asarray(state)[np.newaxis, :]
        new_state = np.asarray(new_state['features'])[np.newaxis, :]

        self.__buffer.store_transition(state, action, reward, new_state)

        if(not self.__buffer.is_valid):
            return
        state, action, reward, new_state = self.__buffer.sample_buffer(self.__batch_size)
        new_critic_values = self.__target_critic.predict(new_state, self.__target_actor.predict(new_state))
        
        target = []
        for j in range(self.__batch_size):
            target.append(reward[j] + self.__gamma*new_critic_values[j])
        target = np.reshape(target, (self.__batch_size,1))

        self.__critic.learn(state, action, target)

        predicted_action = self.__actor.predict(state)
        gradients = self.__critic.get_action_gradients(state, predicted_action)
        self.__actor.learn(state, gradients[0])

        self.__update_network_params()

        self.__learn_step_counter+=1

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
    
    # Generic method to save the current state of the learnt models
    def save_models(self):
        self.__actor.save_checkpoint()
        self.__target_actor.save_checkpoint()
        
        self.__critic.save_checkpoint()
        self.__target_critic.save_checkpoint()

    # Generic method to load existing learnt models
    def load_models(self):
        self.__actor.get_last_checkpoint()
        self.__target_actor.get_last_checkpoint()
        
        self.__critic.get_last_checkpoint()
        self.__target_critic.get_last_checkpoint()
    
class Critic(object):
    def __init__(self, learning_rate, name, session):
        self.__name = name
        self.__session = session
        self.__critic_lr = learning_rate

        # Build the actor network
        self.__build_network()

        # Persisting the model parameters
        self.params = tf.compat.v1.trainable_variables(scope=self.__name)
        self.__saver = tf.compat.v1.train.Saver()
        self.__checkpoint_file = CHECKPOINT_PATH + self.__name + '.ckpt'

        self.__init_policy_grad = tf.gradients(self.__q_value, self.__action_ph)
        self.__optimizer = tf.compat.v1.train.AdamOptimizer(self.__critic_lr).minimize(self.__loss)

    def get_session(self):
        return self.__session

    def __build_network(self):
        with tf.compat.v1.variable_scope(self.__name):
            self.__input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, N_FEATURES], name='inputs')
            self.__action_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, N_ACTIONS], name='actions')
            self.__q_target = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='targets')

            # Input Layer
            f1 = 1/np.sqrt(LAYER_1_NEURONS)
            input_layer = tf.compat.v1.layers.dense(self.__input_ph, units=LAYER_1_NEURONS, 
                                kernel_initializer=RandomUniform(-f1, f1),
                                bias_initializer=RandomUniform(-f1, f1))
            batch_1 = tf.compat.v1.layers.batch_normalization(input_layer)
            input_layer_activation = tf.nn.relu(batch_1)

            # Hidden Layer
            f2 = 1/np.sqrt(LAYER_2_NEURONS)
            hidden_layer = tf.compat.v1.layers.dense(input_layer_activation, units=LAYER_2_NEURONS, 
                                kernel_initializer=RandomUniform(-f2, f2),
                                bias_initializer=RandomUniform(-f2, f2))
            batch_2 = tf.compat.v1.layers.batch_normalization(hidden_layer)

            action_in = tf.compat.v1.layers.dense(self.__action_ph, units=LAYER_2_NEURONS, activation='relu')
            state_actions = tf.add(batch_2, action_in)
            state_actions = tf.nn.relu(state_actions)

            # Output Layer
            f3 = 0.5 # This an intial value
            self.__q_value = tf.compat.v1.layers.dense(state_actions, units=1, 
                                kernel_initializer=RandomUniform(-f3, f3),
                                bias_initializer=RandomUniform(-f3, f3),
                                kernel_regularizer=regularizers.l2(0.01))
            self.__loss = tf.losses.mean_squared_error(self.__q_target, self.__q_value)

    # Get the action
    def predict(self, observations, actions):
        return self.__session.run(self.__q_value, 
                        feed_dict={
                            self.__input_ph : observations,
                            self.__action_ph : actions
                        })

    # Learing the model from experience
    def learn(self, observations, actions, q_target):
        self.__session.run(self.__optimizer, 
                        feed_dict={
                            self.__input_ph : observations,
                            self.__action_ph : actions,
                            self.__q_target : q_target
                        })

    # Get the gradeints for state, action pairs
    def get_action_gradients(self, observations, actions):
        return self.__session.run(self.__init_policy_grad,
                        feed_dict={
                            self.__input_ph : observations,
                            self.__action_ph : actions
                        })

    # Saving the current session in it's current state
    def save_checkpoint(self):
        self.__saver.save(self.__session, self.__checkpoint_file)
    
    # Restoring an existing session
    def get_last_checkpoint(self):
        self.__saver.restore(self.__session, self.__checkpoint_file)

class Actor(object):
    def __init__(self, learning_rate, name, session, action_bound, batch_size):
        self.__name = name
        self.__session = session
        self.__batch_size = batch_size
        self.__actor_lr = learning_rate
        self.__action_bound = action_bound
        
        # Build the actor network
        self.__build_network()

        # Persisting the model parameters
        self.params = tf.compat.v1.trainable_variables(scope=self.__name)
        self.__saver = tf.compat.v1.train.Saver()
        self.__checkpoint_file = CHECKPOINT_PATH + self.__name + '.ckpt'

        # Initializing Policy Gradient
        self.__init_policy_grad = tf.gradients(self.__mu, self.params, self.__action_gradient_ph)
        self.__actor_gradients = list(map(lambda x: tf.divide(x, self.__batch_size), self.__init_policy_grad))
        self.__optimizer = tf.compat.v1.train.AdamOptimizer(self.__actor_lr).apply_gradients(zip(self.__actor_gradients, self.params))
    
    def get_session(self):
        return self.__session

    def __build_network(self):
        with tf.compat.v1.variable_scope(self.__name):
            self.__input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, N_FEATURES], name='inputs')
            self.__action_gradient_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, N_ACTIONS], name='actions')

            # Input Layer
            f1 = 1/np.sqrt(LAYER_1_NEURONS)
            input_layer = tf.compat.v1.layers.dense(self.__input_ph, units=LAYER_1_NEURONS, 
                                kernel_initializer=RandomUniform(-f1, f1),
                                bias_initializer=RandomUniform(-f1, f1))
            batch_1 = tf.compat.v1.layers.batch_normalization(input_layer)
            input_layer_activation = tf.nn.relu(batch_1)

            # Hidden Layer
            f2 = 1/np.sqrt(LAYER_2_NEURONS)
            hidden_layer = tf.compat.v1.layers.dense(input_layer_activation, units=LAYER_2_NEURONS, 
                                kernel_initializer=RandomUniform(-f2, f2),
                                bias_initializer=RandomUniform(-f2, f2))
            batch_2 = tf.compat.v1.layers.batch_normalization(hidden_layer)
            hidden_layer_activation = tf.nn.relu(batch_2)

            # Output Layer
            f3 = 0.5 # This an intial value
            output_layer = tf.compat.v1.layers.dense(hidden_layer_activation, units=N_ACTIONS, 
                                activation='tanh',
                                kernel_initializer=RandomUniform(-f3, f3),
                                bias_initializer=RandomUniform(-f3, f3))
            self.__mu = tf.multiply(output_layer, self.__action_bound)
    
    # Get the action
    def predict(self, observation):
        return self.__session.run(self.__mu, 
                            feed_dict={
                                self.__input_ph : observation
                            })

    # Learing the model from experience
    def learn(self, inputs, gradients):
        self.__session.run(self.__optimizer, 
                        feed_dict={
                            self.__input_ph : inputs,
                            self.__action_gradient_ph : gradients
                        })

    # Saving the current session in it's current state
    def save_checkpoint(self):
        self.__saver.save(self.__session, self.__checkpoint_file)
    
    # Restoring an existing session
    def get_last_checkpoint(self):
        self.__saver.restore(self.__session, self.__checkpoint_file)
