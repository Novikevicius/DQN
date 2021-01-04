from os import error
import numpy as np
import math
import QTable as qt
from QTable import QTable
class DQTable(qt.QTable):
    def __init__(self, action_space, state_space=None, model=None) -> None:
        QTable.__init__(self, action_space, state_space, model)
