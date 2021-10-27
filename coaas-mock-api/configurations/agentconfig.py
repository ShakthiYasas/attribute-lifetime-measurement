from configurations.configuration import Configuration

# Configuration for the DQN Agent
class DQNConfiguration(Configuration):
    explore_mentor = 'Random'

    def __init__(self, config):
        defaults = config['DEFAULT']
        hyper_params = config['HYPER']

        self.optimizer = defaults['Optimizer']

        self.batch_size = int(defaults['MinibatchSize'])
        self.memory_size = int(defaults['MemoryListSize'])
        self.replace_target_iter = int(defaults['ParameterSync'])
        self.dynamic_e_greedy_iter = int(defaults['ExplorationEpoch'])

        self.learning_rate = float(hyper_params['alpha'])
        self.discount_rate = float(hyper_params['gamma'])
        self.e_greedy_init = float(hyper_params['epsilon'])
        self.e_greedy_max = float(hyper_params['epsilon_max'])
        self.e_greedy_increment = float(hyper_params['delta_plus'])
        self.e_greedy_decrement = float(hyper_params['delta_minus'])

        self.history_size = int(hyper_params['history_size'])
        self.reward_threshold = float(hyper_params['reward_max'])

# Configuration for the A3C Agent
class A3CConfiguration(Configuration):
    explore_mentor = 'Random'

    def __init__(self, config):
        defaults = config['DEFAULT']
        hyper_params = config['HYPER']

        self.optimizer = defaults['Optimizer']

        self.batch_size = int(defaults['MinibatchSize'])
        self.memory_size = int(defaults['MemoryListSize'])
        self.replace_target_iter = int(defaults['ParameterSync'])
        self.dynamic_e_greedy_iter = int(defaults['ExplorationEpoch'])

        self.learning_rate = float(hyper_params['alpha'])
        self.discount_rate = float(hyper_params['gamma'])
        self.e_greedy_init = float(hyper_params['epsilon'])
        self.e_greedy_max = float(hyper_params['epsilon_max'])
        self.e_greedy_increment = float(hyper_params['delta_plus'])
        self.e_greedy_decrement = float(hyper_params['delta_minus'])

        self.history_size = int(hyper_params['history_size'])
        self.reward_threshold = float(hyper_params['reward_max'])