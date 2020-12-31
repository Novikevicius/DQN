import numpy as np
from numpy.core.fromnumeric import argmax
import math

class QTable(object):
    def __init__(self, state_space, action_space) -> None:
        self.state_space = state_space
        self.action_space = action_space
        self.table = [Input(self.action_space) for i in range(self.state_space)]
    def __str__(self):
        s = ""
        for input in self.table:
            s += str(input) + "\n"
        return s

class Input(object):
    def __init__(self, min, max, step_size, precision=2) -> None:
        self.precision = precision
        self.min = min
        self.max = max
        self.step_size = step_size
        self.size = int((self.max-self.min) / step_size + 1) + 2 # -inf & +inf
        self.values = [0] * self.size
    def split(self, step_size):
        return (self.max - self.min) / step_size
    def map(self, value):
        if( value < self.min):
            return 0
        if (value > self.max):
            return self.size-1
        rounded = round(value, self.precision)
        t = rounded - self.min
        t2 = math.ceil(t / self.step_size)
        return t2 + 1


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
