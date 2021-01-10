from os import error
import numpy as np
import math

class QTable(object):
    def __init__(self, action_space, state_space=None, model=None) -> None:
        if model:
            self.table = model
            self.state_space = len(model)
        else:
            self.table = [Input(0, 1, 0.1, 1) for i in range(self.state_space)]
            self.state_space = state_space
        self.shape = [self.table[i].size for i in range(self.state_space)]
        self.action_space = action_space
        self.shape.append(action_space)
        self.shape = tuple(self.shape)
        self.n = action_space
        for i in range(self.state_space):
            self.n *= self.table[i].size
        self.values = np.array([0.0] * self.n)
        self.values = np.reshape(self.values, self.shape)
        
        
    def __str__(self):
        return str(self.values)
    def getValue(self, state):
        indexes = []
        for i in range(self.state_space):
            if type(state) is list:
                indexes.append(self.table[i].map(state[i]))
            else:
                indexes.append(self.table[i].map(state))
        return self.values[tuple(indexes)]

    
    def setValue(self, state, action, value):
        indexes = []
        for i in range(self.state_space):
            if type(state) is list:
                indexes.append(self.table[i].map(state[i]))
            else:
                indexes.append(self.table[i].map(state))
        indexes.append(action)
        t = self.values[tuple(indexes)]
        self.values[tuple(indexes)] = value
        t = self.values[tuple(indexes)]
        t = 0
    
    def save(self, fileName):
        a = np.asfarray(self.values)
        np.save(fileName, a)
        with open(fileName + '.inputs', 'w') as f:
            f.write('action_space: ' + str(self.action_space) + '\n')
            f.write('state_space: ' + str(self.state_space) + '\n')
            for i in self.table:
                i.save(f)
    @staticmethod
    def load(fileName):
        values = np.array(np.load(fileName + '.npy'))
        model = []
        with open(fileName + '.inputs', 'r') as f:
            action_space = int(QTable.read_field(f, 'action_space'))
            state_space  = int(QTable.read_field(f, 'state_space'))
            for i in range(state_space):
                model.append(Input.load(f))
        table = QTable(action_space, state_space, model)
        table.values = values
        return table

    @staticmethod
    def read_field(f, field_name):
        line = f.readline()
        tokens = line.splitlines()[0].split(' ')
        if len(tokens) != 2:
            raise error("Wrong file format: not enough tokens, expected: 2, got: " + str(len(tokens)))
        if tokens[0] != field_name + ':':
            raise error("Wrong file format: expected: " + field_name + ", got: " + tokens[0])
        return tokens[1]

class Input(object):
    def __init__(self, min, max, step_size, precision=2, index=0, callback=None) -> None:
        self.precision = precision
        self.min = min
        self.max = max
        self.step_size = step_size
        self.size = int((self.max-self.min) / step_size + 1) + 1 # -inf & +inf
        self.index = index
        self.callback = callback
        self.indexes = [(self.min + i * self.step_size, 0) for i in range(self.size-1)]
        self.indexes.append((float('+inf'), 0))
    def split(self, step_size):
        return (self.max - self.min) / step_size
    def map(self, value, count_hits = False):
        MIN_HITS_BEFORE_SPLIT = 5
        for i in range(len(self.indexes)):
            v, hits = self.indexes[i]
            if value <= v:
                self.indexes[i] = (v, hits+1)
                if i == 0 or i == self.size-1:
                    mid = self.step_size
                else:
                    mid = (v-self.indexes[i-1][0]) / 2
                if(count_hits and hits+1 >= MIN_HITS_BEFORE_SPLIT):
                    self.indexes[i] = (v, 0)
                    if i == self.size-1:
                        self.indexes.insert(i, (self.indexes[i-1][0]+mid, 0))
                    else:
                        self.indexes.insert(i, (v-mid, 0))
                    self.size += 1
                    if self.callback:
                        self.callback(self.index, i)
                return i
    def __str__(self):
        s = '['
        for i in range(self.size-2):
            s += str(self.values[i]) + ', '
        s += str(self.values[self.size-1]) + ']'
        return s
    def save(self, f):
        f.write(str(self.precision) + '\n')
        f.write(str(self.min) + '\n')
        f.write(str(self.max) + '\n')
        f.write(str(self.step_size) + '\n')
    
    @staticmethod
    def load(file):
        precision = int(file.readline().splitlines()[0])
        min = float(file.readline().splitlines()[0])
        max = float(file.readline().splitlines()[0])
        step_size = float(file.readline().splitlines()[0])
        return Input(min, max, step_size, precision)




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
