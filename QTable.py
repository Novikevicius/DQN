from os import error
from tokenize import Number
import numpy as np

class QTable(object):
    def __init__(self, action_space, state_space=None, model=None, dynamic=False) -> None:
        if model:
            self.table = model
            self.state_space = len(model)
            for i in range(self.state_space):
                self.table[i].index = i
                self.table[i].callback = self.split
        else:
            self.table = [Input(0, 1, 0.1, 1, index=i, callback=self.split) for i in range(self.state_space)]
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
        self.dynamic = dynamic
        
    def split(self, index, i):
        self.shape = [self.table[i].size for i in range(index)]
        self.values = np.insert(self.values, i, np.take(self.values, i, axis=index), axis=index)
    def __str__(self):
        return str(self.values)
    def get_intervals(self):
        intervals = []
        for input in self.table:
            intervals.append(input.get_intervals())
        return intervals
    def getValue(self, state, count_hits=False):
        indexes = []
        for i in range(self.state_space):
            if type(state) is list or type(state) is np.array or type(state) is np.ndarray:
                indexes.append(self.table[i].map(state[i], count_hits))
            else:
                indexes.append(self.table[i].map(state, count_hits))
        return self.values[tuple(indexes)]

    
    def setValue(self, state, action, value, count_hits=False):
        count_hits = count_hits or self.dynamic
        indexes = []
        for i in range(self.state_space):
            if type(state) is list or type(state) is np.array or type(state) is np.ndarray:
                indexes.append(self.table[i].map(state[i], False))
            else:
                indexes.append(self.table[i].map(state, False))
        indexes.append(action)

        diff = abs(value-self.values[tuple(indexes)])
        split = diff > 0.01

        self.values[tuple(indexes)] = value
        if self.dynamic:
            for i in range(self.state_space):
                if type(state) is list or type(state) is np.array or type(state) is np.ndarray:
                    self.table[i].map(state[i], count_hits, split)
                else:
                    self.table[i].map(state, count_hits, split)

    # get Out Of Bounds step sizes
    def get_OOB(self):
        return [inpt.OUT_OF_BOUNDS_STEP_SIZE for inpt in self.table]
    
    def save(self, fileName, static=True):
        a = np.asfarray(self.values)
        np.save(fileName, a)
        with open(fileName + '.inputs', 'w') as f:
            f.write('static: ' + str(static) + '\n')
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
    def __init__(self, min, max, step_size, precision=2, index=0, callback=None, min_hits=25, static=True, out_of_bounds_step=None) -> None:
        self.precision = precision
        self.min = min
        self.max = max
        self.step_size = step_size
        self.size = int((self.max-self.min+1) / step_size + 1) + 1 # -inf & +inf
        self.index = index
        self.callback = callback
        self.indexes = [(self.min + i * self.step_size, 0) for i in range(self.size-1)]
        self.indexes.append((float('+inf'), 0))
        #self.EPSILON = float('0.' + ('0' * (self.precision-1)) +'1')
        self.EPSILON = 0.1
        #self.min_hits = min_hits
        self.min_hits = 1
        self.static = static
        self.OUT_OF_BOUNDS_STEP_SIZE = self.step_size if out_of_bounds_step == None else out_of_bounds_step
        
    def split(self, step_size):
        return (self.max - self.min) / step_size
    def map(self, value, count_hits = False, split=True):
        for i in range(len(self.indexes)):
            v, hits = self.indexes[i]
            if value <= v:
                if self.static:
                    return i
                self.indexes[i] = (v, hits+1)
                if i == 0 or i == self.size-1:
                    mid = self.OUT_OF_BOUNDS_STEP_SIZE
                    #mid = self.step_size
                    if i == 0:
                        diff = v - self.indexes[i+1][0]
                    else:
                        diff = v - self.indexes[i-1][0]
                else:
                    diff = v - self.indexes[i-1][0]
                    mid = (v-self.indexes[i-1][0]) / 2
                if abs(value - v) < self.EPSILON or abs(diff) < self.EPSILON or not split:
                    return i
                if(count_hits and hits+1 >= self.min_hits):
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
            s += str(self.indexes[i]) + ', '
        s += str(self.indexes[self.size-1]) + ']'
        return s
    def get_intervals(self):
        s = [float('-inf')]
        for i in range(self.size-1):
            s.append(self.indexes[i][0])
        s.append(float('+inf'))
        return s

    def save(self, f):
        f.write(str(self.precision) + '\n')
        f.write(str(self.min) + '\n')
        f.write(str(self.max) + '\n')
        f.write(str(self.step_size) + '\n')
        f.write(str(self.size) + '\n')
        f.write(str(len(self.indexes)) + '\n')
        i = 0
        for (v, hits) in self.indexes:
            f.write( str(i) + ' ' + str(v) + ' ' + str(hits) + ' ')
            i += 1
    
    @staticmethod
    def load(file):
        precision = int(file.readline().splitlines()[0])
        min = float(file.readline().splitlines()[0])
        max = float(file.readline().splitlines()[0])
        step_size = float(file.readline().splitlines()[0])
        return Input(min, max, step_size, precision)
