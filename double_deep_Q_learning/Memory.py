import collections


class Memory:
    def __init__(self, size):
        self.size = size
        self.deque = collections.deque([])  # todo this or
        self.queue = []                     # todo this

    def sample(self):
        """Returns a sample of the memory"""
        pass

    def record(self):
        """Append new memory to the memory list"""
        pass
