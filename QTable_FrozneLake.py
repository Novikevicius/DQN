import numpy as np
from numpy.core.fromnumeric import argmax
import math

class QTable(object):
    def __init__(self, state_space, action_space) -> None:
        self.state_space = 1
        self.action_space = action_space
        self.table = [Input(0, 16, 1, 0) for i in range(self.state_space)]
        self.shape = [self.table[i].size for i in range(self.state_space)]
        self.shape.append(action_space)
        self.shape = tuple(self.shape)
        self.n = action_space
        for i in range(self.state_space):
            self.n *= self.table[i].size
        self.values = np.reshape(np.zeros(self.n), self.shape)
        
        
    def __str__(self):
        return str(self.values)
        s = ""
        for input in self.table:
            s += str(input) + "\n"
        return s
    def getValue(self, state):
        indexes = []
        for i in range(self.state_space):
            indexes.append(self.table[i].map(state))
        return self.values[tuple(indexes)]
        v = self.values
        for i in range(self.state_space):
            v = v[self.table[i].map(state[i])]
        return v
    def printNonZeros(self):
        for x1 in self.values:
            for x2 in x1:
                for x3 in x2:
                    for v in x3:
                        if v[0] != 0 or v[1] != 0:
                            print(v, ',')

    
    def setValue(self, state, action, value):
        indexes = []
        for i in range(self.state_space):
            indexes.append(self.table[i].map(state))
        indexes.append(action)
        self.values[tuple(indexes)] = value

class Input(object):
    def __init__(self, min, max, step_size, precision=2) -> None:
        self.precision = precision
        self.min = min
        self.max = max
        self.step_size = step_size
        self.size = int((self.max-self.min) / step_size + 1) + 2 # -inf & +inf
        #self.values = [0] * self.size
    def split(self, step_size):
        return (self.max - self.min) / step_size
    def map(self, value):
        if( value < self.min):
            return 0
        if (value > self.max):
            return self.size-1
        return math.ceil((round(value, self.precision) - self.min) / self.step_size) + 1
    def getValue(self, x):
        return self.values[self.map(x)]
    def setValue(self, x, new_value):
        self.values[self.map(x)] = new_value
    def increaseValueBy(self, x, value):
        self.values[self.map(x)] += value
    def __str__(self):
        s = '['
        for i in range(self.size-2):
            s += str(self.values[i]) + ', '
        s += str(self.values[self.size-1]) + ']'
        return s


def test_map():
    min = 1
    max = 5
    step = 1
    input = Input(min, max, step, 0)
    assert(input.map(min) == 1)
    assert(input.map(max) == input.size-2)
    assert(input.map(min-1) == 0)
    assert(input.map(max+1) == input.size-1)
    assert(input.map(1.5) == input.map(2))
    for i in range(min, max, step):
        assert(input.map(i) == i)
    min = -1
    max = 3
    step = 0.1
    input = Input(min, max, step, 1)
    assert(input.map(min) == 1)
    assert(input.map(max) == input.size-2)
    assert(input.map(min-1) == 0)
    assert(input.map(max+1) == input.size-1)
    assert(input.map(min+step) == input.map(min) + 1)
    assert(input.map(min+step*5) == input.map(min) + 5)
