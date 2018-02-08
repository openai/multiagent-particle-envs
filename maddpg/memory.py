from collections import namedtuple
import random
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =[]
        self.position = 0

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)












