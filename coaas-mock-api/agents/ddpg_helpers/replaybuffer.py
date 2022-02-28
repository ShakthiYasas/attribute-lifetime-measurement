import numpy as np

class ReplayBuffer():
    # This allows manipulation of datatypes that using a Dequeue instead would not allow. 
    # So, this is a memory management strategy.
    def __init__(self, max_size, batch_size, input_shape, n_actions):
        self.mem_cntr = 0 # position of the most recently saved memory
        self.is_valid = False
        self.mem_size = max_size
        self.batch_size = batch_size

        # State, Action, Reward, New State buffer memories
        self.reward_memory = np.zeros(self.mem_size)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))

    def store_transition(self, state, action, reward, new_state):
        # This acts like a FIFO Queue.
        index = self.mem_cntr % self.mem_size 

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state

        self.mem_cntr += 1
        if(self.mem_cntr >= self.batch_size):
            self.is_valid = True

    def sample_buffer(self):
        max_mem = 0 
        if(self.mem_cntr < self.mem_size):
            max_mem = self.mem_cntr
        else:
            max_mem = self.mem_size

        idx = np.random.randint(max_mem, size=self.batch_size)

        states = self.state_memory[idx]
        actions = self.action_memory[idx]
        rewards = self.reward_memory[idx]
        new_states = self.new_state_memory[idx]

        return states, actions, rewards, new_states

    def get_entire_buffer(self):
        states = self.state_memory
        actions = self.action_memory
        rewards = self.reward_memory
        new_states = self.new_state_memory

        return states, actions, rewards, new_states
    
    