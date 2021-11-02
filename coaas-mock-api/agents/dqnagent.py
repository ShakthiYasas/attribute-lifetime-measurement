from agent import Agent

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from exploreagent import RandomAgent, MRUAgent, MFUAgent

# Random seeding
np.random.seed(1)
tf.set_random_seed(1)

# Disabling eager execution
tf.disable_eager_execution()

# DQN based RL agent
class DQNAgent(Agent):
    def __init__(self, config, caller):
        # General Configurations
        self.__caller = caller
        self.__memory_size = config.memory_size
        self.__history_size = config.history_size  
        self.__reward_threshold = config.reward_threshold
        self.__replace_target_iter = config.replace_target_iter
        self.__batch_size = max(config.batch_size, int(config.memory_size*0.25))

        # Setting the hyper-parameters
        self.__learningrate = config.learning_rate 
        self.__gamma = config.discounting_factor
        self.__epsilons_max = config.e_greedy_max
        self.__epsilons_increment = config.e_greedy_increment
        self.__epsilons_decrement = config.e_greedy_decrement

        # Get the initial action space from offline processing
        # self.action_space.append()

        # Get the initial state space from offline processing
        # self.__init_state_space = 
        # i.e. (This need to be modified as self.__init_state_space.size - size being a property of _init_state_space)
        self.__init_state_space_size = 50

        # Setting the features
        self.__feature_vector = [
            'short_term_access', 'expected_short_term_access', 
            'mid_term_access', 'expected_mid_term_access',
            'long_term_access', 'expected_long_term_access',            
            'average_cached_lifetime', 'expected_marginal_utility'
            'short_term_hitrate', 'expected_short_term_hitrate',
            'mid_term_hitrate', 'expected_mid_term_hitrate',
            'long_term_hitrate', 'expected_long_term_hitrate'
            ]
        self.__n_features = len(self.__feature_vector)*self.__init_state_space_size

        
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
        
        # Initializing Re-play memory
        # Initializing Zero Memory Status [state, action, reward, new_state]
        self.__cost_his = []
        self.__memory_counter = 0
        self.__reward_history = []    
        self.__memory = np.zeros((self.__memory_size, self.__n_features * 2 + 2))
        
        # Accumalated count of learning epochs 
        self.__learn_step_counter = 0
        
        # Building the Deep Q Network containing [target_net, evaluate_net] 
        # and start the session
        self.__build_network(config.optimizer)
        self.__replace_target_op = [tf.assign(t, e) for t, e in zip(tf.get_collection('target_net_params'), 
                                    tf.get_collection('eval_net_params'))]
        self.__sess = tf.Session()
        tf.summary.FileWriter("logs/", self.__sess.graph)
        self.__sess.run(tf.global_variables_initializer())
        
    def __build_network(self, optimizer:str):
        # Cleaning the DQ Network (Reset to initial state)
        tf.reset_default_graph()
        
        # Building the Evaluation Network
        # Inputs
        self.state = tf.placeholder(tf.float32, [None, self.__n_features], name='s')  
        # For calculating losses (Target Q-value)
        self.q_target = tf.placeholder(tf.float32, [None, len(self.action_space)], name='Q_target') 
        with tf.variable_scope('eval_net'):
            # Configuration parameters of the of layers
            # c_names(collections_names) are the collections to store variables
            n_l1, n_l2 = 64, 32
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) 
                
            # First layer. Collections is used later when assigning to target network
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.__n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            # Second layer. Collections is used later when assigning to target network
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # Output layer. Collections is used later when assigning to target network
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, len(self.action_space)], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, len(self.action_space)], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        # Evaluating the root mean sqaure error between the output and the target Q-Values
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            if(optimizer.lower() == 'rmsprop'):
                self._train_op = tf.train.RMSPropOptimizer(self.__learningrate).minimize(self.loss)
            elif(optimizer.lower() == 'adam'):
                self._train_op = tf.train.AdamOptimizer(learning_rate=self.__learningrate).minimize(self.loss)

        # Building the Target Network
        # Inputs
        self.new_state = tf.placeholder(tf.float32, [None, self.__n_features], name='s_') 
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # First layer. Collections is used later when assigning to target network
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.__n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.new_state, w1) + b1)

            # Second layer. Collections is used later when assigning to target network
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # Output layer. Collections is used later when assigning to target network
            with tf.variable_scope('32'):
                w3 = tf.get_variable('w2', [n_l2, len(self.action_space)], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, len(self.action_space)], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    # Update the transition probabilities
    def store_transition(self, state, action, reward, new_state):
        # Here, the old and the new state should be mapped to existing cluster 
        # or develop a new cluster
        transition = np.hstack((state, [action, reward], new_state))

        # Replace the old memory with new memory
        index = self.__memory_counter % self.__memory_size
        self.__memory[index, :] = transition
        self.__memory_counter += 1
        
        # Record reward
        if(len(self.__reward_history) == self.__history_size):
            self.__reward_history.pop(0)
        self.__reward_history.append(reward)

    # Select the most suitable action given a state
    # The biggest challenge here is the change of state with changes in the environment. 
    # So, there is no gurantee that the state will remain unchanged. 
    # So, the only option is to find the state that is the most similar using clustering.
    def choose_action(self, observation):
        random_value = np.random.uniform()
        if(random_value < self.__epsilons):
            if(isinstance(self.__explore_mentor,MFUAgent)):
                action = self.__explore_mentor.choose_action(self.__caller.get_attribute_access_trend())
            else:
                action = self.__explore_mentor.choose_action(self.__caller.get_observed())
                # Here the action is a (entityid,attribute) to cache
        else:
            observation = observation[np.newaxis, :]
            # Feed forwarding the observations to retrieve Q value for every actions
            actions_value = self.__sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(actions_value)
            # Translate this index to the (entityid,attribute) pair
            
        return action

    # Learn the network
    def learn(self):
        # Checking to replace target parameters
        if self.__learn_step_counter % self.__replace_target_iter == 0:
            self.__sess.run(self.__replace_target_op)

        # Sampling batch memory from all memory instances
        if self.__memory_counter > self.__memory_size:
            sample_index = np.random.choice(self.__memory_size, size=self.__batch_size)
        else:
            sample_index = np.random.choice(self.__memory_counter, size=self.__batch_size)
        batch_memory = self.__memory[sample_index, :]

        q_next, q_eval = self.__sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                 # Fixed parameteres
                self.new_state: batch_memory[:, -self.__n_features:], 
                # New parameteres
                self.state: batch_memory[:, :self.__n_features],  
            })

        # Updating the Q_Target in relation to Q_Eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.__batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.__n_features].astype(int)
        reward = batch_memory[:, self.__n_features + 1]

        # Q-Value update
        q_target[batch_index, eval_act_index] = reward + self.__gamma * np.max(q_next, axis=1)

        # Training the Evaluation Network
        _, self.cost = self.__sess.run([self._train_op, self.loss],
            feed_dict={self.state: batch_memory[:, :self.__n_features], self.q_target: q_target}
        )
        self.__cost_his.append(self.cost)

        # Increasing or Decreasing epsilons
        if self.__learn_step_counter % self.__dynamic_e_greedy_iter == 0:

            # If e-greedy
            if self.__epsilons_decrement is not None:
                # Dynamic bidirectional e-greedy 
                # That allows the netowrk to be self adaptive between exploration and exploitation depending on the
                # current performance of the system. i.e., if MR is high, the increase epsilon (more exploration)
                # and on the contrary, decrease epsilon if MR is less (more exploitation). 
                if self.__epsilons_increment is not None:
                    rho = np.median(np.array(self.__reward_history))
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

        self.__learn_step_counter += 1

    def get_feature_vector(self):
        return self.__feature_vector

    def plot_cost(self):
        plt.plot(np.arange(len(self.__cost_his)), self.__cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()