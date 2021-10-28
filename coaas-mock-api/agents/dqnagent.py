from agent import Agent

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from exploreagent import RandomAgent, MRUAgent, MFUAgent

np.random.seed(1)
tf.set_random_seed(1)

# Disabling eager execution
tf.disable_eager_execution()

class DQNAgent(Agent):
    def __init__(self, config):
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.gamma = config.discounting_factor

        self.epsilons_min = config.e_greedy_min
        self.epsilons_max = config.e_greedy_max
        self.epsilons_increment = config.e_greedy_increment
        self.epsilons_decrement = config.e_greedy_decrement
        
        self.epsilons = list(config.e_greedy_init)
        if (config.e_greedy_init is None) or (config.e_greedy_decrement is None):
            self.epsilons = list(self.epsilons_min)
        
        # Selecting the algorithem to execute during the exploration phase
        self.explore_mentor = None
        if config.explore_mentor.lower() == 'mru':
            self.explore_mentor = MRUAgent()
        elif config.explore_mentor.lower() == 'mfu':
            self.explore_mentor = MFUAgent()
        else:
            self.explore_mentor = RandomAgent()
        
        self.replace_target_iter = config.replace_target_iter
        self.memory_size = config.memory_size

        # Accumilated learning epochs 
        self.learn_step_counter = 0

        # Initializing zero memory [s, a, r, s_]
        ########################################## This need to fixed
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        
        # Initializing the history set of Rewards
        self.reward_history = []
        self.history_size = config.history_size
        self.dynamic_e_greedy_iter = config.dynamic_e_greedy_iter
        self.reward_threshold = config.reward_threshold

        # Building the Deep Q Network containing [target_net, evaluate_net]
        self.__build_network(config.optimizer)
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        # Output graph
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        
        self.verbose = 0

    def __build_network(self, optimizer:str):
        # Cleaning the DQ Network (Reset to initial state)
        tf.reset_default_graph()
        
        # Building the Evaluation Network
        # Inputs
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  
         # For calculating losses
        self.q_target = tf.placeholder(tf.float32, [None, len(self.action_space)], name='Q_target') 
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, 32, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # Configuration of the of layers

            # First layer. Collections is used later when assigning to target network
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

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

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            if(optimizer.lower() == 'rmsprop'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            elif(optimizer.lower() == 'adam'):
                self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Building the Target Network
        # Inputs
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') 
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # First layer. Collections is used later when assigning to target network
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # Second layer. Collections is used later when assigning to target network
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # Output layer. Collections is used later when assigning to target network
            with tf.variable_scope('32'):
                w3 = tf.get_variable('w2', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    # Update the transition probabilities
    def store_transition(self, state, action, reward, new_state):
        state, new_state = state['features'], new_state['features']
        transition = np.hstack((state, [action, reward], new_state))

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
        # Record reward
        if len(self.reward_history) == self.history_size:
            self.reward_history.pop(0)
        self.reward_history.append(reward)

    # Select the most suitable action given a state
    # The biggest challenge here the change of state with changes in the environment. 
    # So, there is no gurantee that the state will remain unchanged. 
    # So, the only option is to find the state that is the most similar.

    def choose_action(self, observation):
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        elif self.epsilons[0] <= coin and coin < self.epsilons[0] + self.epsilons[1]:
            action = self.explore_mentor._choose_action(observation)
        else:
            observation = observation['features']
            observation = observation[np.newaxis, :]

            # Feed forwarding the observations to retrieve Q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            
        if action < 0 or action > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % action)
            
        return action

    # Learn the network
    def learn(self):
        # Checking to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # Verbose check
            if self.verbose >= 1:
                print('Target DQN Parameters Replaced')

        # Sampling batch memory from all memory instances
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                 # Fixed parameteres
                self.s_: batch_memory[:, -self.n_features:], 
                # New parameteres
                self.s: batch_memory[:, :self.n_features],  
            })

        # Updating the Q_Target in relation to Q_Eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Training the Evaluation Network
        _, self.cost = self.sess.run([self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target}
        )
        self.cost_his.append(self.cost)

        # Verbose                    
        if (self.verbose == 2 and self.learn_step_counter % 100 == 0) or \
            (self.verbose >= 3 and self.learn_step_counter % 20 == 0):
            print("Step=%d: Cost=%d" % (self.learn_step_counter, self.cost))

        # Increasing or Decreasing epsilons
        if self.learn_step_counter % self.dynamic_e_greedy_iter == 0:

            # If e-greedy
            if self.epsilons_decrement is not None:
                # Dynamic bidirectional e-greedy 
                # That allows the netowrk to be self adaptive between exploration and exploitation depending on the
                # current performance of the system. i.e., if MR is high, the increase epsilon (more exploration)
                # and on the contrary, decrease epsilon if MR is less (more exploitation). 
                if self.epsilons_increment is not None:
                    rho = np.median(np.array(self.reward_history))
                    if rho >= self.reward_threshold:
                        self.epsilons[0] -= self.epsilons_decrement[0]
                        self.epsilons[1] -= self.epsilons_decrement[1]
                        # Verbose
                        if self.verbose >= 3:
                            print("Eps down: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                    else:
                        self.epsilons[0] += self.epsilons_increment[0]
                        self.epsilons[1] += self.epsilons_increment[1]
                        # Verbose                    
                        if self.verbose >= 3:
                            print("Eps up: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                
                # Traditional e-greedy
                else:
                    self.epsilons[0] -= self.epsilons_decrement[0]
                    self.epsilons[1] -= self.epsilons_decrement[1]

            # Enforce upper bound and lower bound
            truncate = lambda x, lower, upper: min(max(x, lower), upper)
            self.epsilons[0] = truncate(self.epsilons[0], self.epsilons_min[0], self.epsilons_max[0])
            self.epsilons[1] = truncate(self.epsilons[1], self.epsilons_min[1], self.epsilons_max[1])

        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()