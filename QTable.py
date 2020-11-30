import numpy as np
from numpy.core.fromnumeric import argmax

class QTable(object):
    def __init__(self, state_space, action_space) -> None:
        self.state_space = state_space
        self.action_space = action_space
        b1, b2 = Bucket(self.action_space, float('-inf'), float('inf')).split()
        self.table = [b1, b2]
    def train(self, env, epochs=100, lr=0.01):
        pass
    def get_values(self, state):
        for b in self.table:
            if b.is_in_range:
                return b
        return None
    def get_max_value(self, state):
        return self.get_value(state).get_max()
    def get_best_action(self, state):
        pass #return
    def split(self, i, step=100):
        if i < 0 or i > len(self.table):
            return
        b1, b2 = self.table[i].split(step)
        a1 = self.table[:(i-1)]
        a2 = self.table[(i+1):]
        self.table = a1
        self.table.append(b1)
        self.table.append(b2)
        self.table.extend(a2)
    def __str__(self):
        s = "["
        for b in self.table:
            s += '{' + str(b) + "},\n"
        s += ']'
        return s
class Bucket(object):
    # [min, max)
    def __init__(self, space, min, max, isTraining=True):
        self.space = space
        self.min = min
        self.max = max
        self.values = np.zeros(self.space)
        self.occurences = 0
    def get_value(self, index):
        if index < 0 or index > self.space:
            return
        return self.values[index]
    def get_max(self):
        return (max(self.values), argmax(self.values))
    def set_value(self, index, value):
        if index < 0 or index > self.space:
            return
        if value < min or value >= max:
            return
        self.values[index] = value
    def resize(self, min, max):
        self.min = min
        self.max = max
    def split(self, step=100):
        min = self.min
        max = self.max
        if min == float('-inf') and max == float('inf'):
            mid = 0
        elif min == float('-inf'):
            mid = max - step
        elif max == float('inf'):
            mid = min + step
        else:            
            mid = (self.max+self.min)/2
        b1 = Bucket(self.space, min, mid)
        b1.values = np.copy(self.values)
        b2 = Bucket(self.space, mid, max)
        b2.values = np.copy(self.values)
        return (b1, b2)
    def is_in_range(self, value):
        return value >= self.min and value < self.max
    def __str__(self):
        return str.format("[{0}, {1}) ", self.min, self.max) + str(self.values)