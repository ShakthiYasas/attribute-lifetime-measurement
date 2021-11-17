from configurations.configuration import Configuration

# Configuration for the A3C Agent
class SimpleConfiguration(Configuration):
    def __init__(self, config):
        defaults = config['DEFAULT']
        hyper_params = config['HYPER']

        self.window = int(defaults['MovingWindow'])
        self.discount_rate = float(hyper_params['Gamma'])
        self.short = int(defaults['ShortWindow'])
        self.mid = int(defaults['MidWindow'])
        self.long = int(defaults['LongWindow'])

        self.explore_mentor = defaults['ExplorationAlgorithm']
        self.e_greedy_init = float(hyper_params['Epsilon'])
        self.e_greedy_max = float(hyper_params['Epsilon_max'])
        self.e_greedy_increment = float(hyper_params['Delta_plus'])
        self.e_greedy_decrement = float(hyper_params['Delta_minus'])

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

        self.learning_rate = float(hyper_params['Alpha'])
        self.discount_rate = float(hyper_params['Gamma'])
        self.e_greedy_init = float(hyper_params['Epsilon'])
        self.explore_mentor = defaults['ExplorationAlgorithm']
        self.e_greedy_max = float(hyper_params['Epsilon_max'])
        self.e_greedy_increment = float(hyper_params['Delta_plus'])
        self.e_greedy_decrement = float(hyper_params['Delta_minus'])

        self.history_size = int(defaults['HistorySize'])
        self.reward_threshold = float(defaults['MaxReward'])

# Configuration for the Actor Critic Agent
class ACConfiguration(Configuration):
    explore_mentor = 'Random'

    def __init__(self, config):
        defaults = config['DEFAULT']
        hyper_params = config['HYPER']

        self.optimizer = defaults['Optimizer']

        self.learning_rate = float(hyper_params['Alpha'])
        self.discount_rate = float(hyper_params['Gamma'])
        self.e_greedy_init = float(hyper_params['Epsilon'])
        self.explore_mentor = defaults['ExplorationAlgorithm']
        self.e_greedy_max = float(hyper_params['Epsilon_max'])
        self.e_greedy_increment = float(hyper_params['Delta_plus'])
        self.e_greedy_decrement = float(hyper_params['Delta_minus'])
        self.dynamic_e_greedy_iter = int(defaults['ExplorationEpoch'])

        self.reward_threshold = float(defaults['MaxReward'])

        self.pair_similarity_maximum = float(defaults['MaxPairSimilarity'])
        self.cluster_similarity_threshold = float(defaults['ClusterSimilarityThreshold'])
        self.subcluster_similarity_threshold = float(defaults['SubClusterSimilarityThreshold'])

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

        self.learning_rate = float(hyper_params['Alpha'])
        self.discount_rate = float(hyper_params['Gamma'])
        self.e_greedy_init = float(hyper_params['Epsilon'])
        self.explore_mentor = defaults['ExplorationAlgorithm']
        self.e_greedy_max = float(hyper_params['Epsilon_max'])
        self.e_greedy_increment = float(hyper_params['Delta_plus'])
        self.e_greedy_decrement = float(hyper_params['Delta_minus'])

        self.history_size = int(defaults['HistorySize'])
        self.reward_threshold = float(defaults['MaxReward'])

        self.pair_similarity_maximum = float(defaults['MaxPairSimilarity'])
        self.cluster_similarity_threshold = float(defaults['ClusterSimilarityThreshold'])
        self.subcluster_similarity_threshold = float(defaults['SubClusterSimilarityThreshold'])

# Configuration for the Deep Detterministic Policy Gradient Actor Critic Agent
class DDPGConfiguration(Configuration):
    explore_mentor = 'Random'

    def __init__(self, config):
        defaults = config['DEFAULT']
        hyper_params = config['HYPER']

        self.optimizer = defaults['Optimizer']

        self.batch_size = int(defaults['MinibatchSize'])
        self.memory_size = int(defaults['MemoryListSize'])
        self.replace_target_iter = int(defaults['ParameterSync'])
        self.dynamic_e_greedy_iter = int(defaults['ExplorationEpoch'])

        self.learning_rate = float(hyper_params['Alpha'])
        self.discount_rate = float(hyper_params['Gamma'])
        self.e_greedy_init = float(hyper_params['Epsilon'])
        self.explore_mentor = defaults['ExplorationAlgorithm']
        self.e_greedy_max = float(hyper_params['Epsilon_max'])
        self.e_greedy_increment = float(hyper_params['Delta_plus'])
        self.e_greedy_decrement = float(hyper_params['Delta_minus'])

        self.history_size = int(defaults['HistorySize'])
        self.reward_threshold = float(defaults['MaxReward'])

        self.pair_similarity_maximum = float(defaults['MaxPairSimilarity'])
        self.cluster_similarity_threshold = float(defaults['ClusterSimilarityThreshold'])
        self.subcluster_similarity_threshold = float(defaults['SubClusterSimilarityThreshold'])