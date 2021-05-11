from Agent import QT_Agent
import DQN
from QTable import *

from functools import *
from datetime import datetime
import os

import QTable

def main():
    input = Input(-1, 1, 1, 0.1, static=False)
    r = input.map(0)
    print(input, r)
    print(input.get_intervals())
    '''
    qt_exp = Experiment(id=9999, agent_name='QT_Agent')
    qt_agent = QT_Agent('CartPole-v1', model=[QTable.Input(-1, 1, 0.1, 4), 
             QTable.Input(-1, 1, 0.1, 4),
             QTable.Input(-1, 1, 0.1, 4),
             QTable.Input(-1, 1, 0.1, 4)])
    qt_exp.run(qt_agent, params={'epochs': 10000, 'lr': 0.5, 'gamma': 0.99, 'reward_function':sum})
    qt_exp.run(qt_agent, params={'epochs': 10000, 'lr': 0.1, 'gamma': 0.99})
'''
    '''
    frozen_lake_qt_exp = Experiment(id=9999, agent_name='QT_Agent')
    frozen_lake_qt_agent = QT_Agent('FrozenLake-v0', model=[QTable.Input(0, 15, 1, 1)])
    frozen_lake_qt_exp.run(frozen_lake_qt_agent, params={'epochs': 10000, 'lr': 0.9, 'gamma': 0.99, 'reward_function':max})
    frozen_lake_qt_exp.run(frozen_lake_qt_agent, params={'epochs': 10000, 'lr': 0.05, 'gamma': 0.99})
    '''
    '''
    dqn_exp = Experiment(id=9998, agent_name='DQN_Agent')
    dqn_agent = DQN.DQN_Agent('CartPole-v1', 9998, params={'activation':'linear', 'loss':'mse', 'lr':0.01})
    dqn_exp.run(dqn_agent, params={'epochs':1000})
    '''
    Input
    


class Experiment:
    experiments_folder = 'experiments'
    def __init__(self, id, agent_name) -> None:
        self.id = id 
        self.agent_name = agent_name
        self.folder = self.experiments_folder + '/' + self.agent_name   

        #create folders to store experiment's data
        if not os.path.exists(self.experiments_folder):
            os.mkdir(self.experiments_folder)
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def __makeFile(self, env_name):
        env_folder = self.folder + '/' + env_name
        if not os.path.exists(env_folder):
            os.mkdir(env_folder)

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        file_name       = dt_string + '_ID_' + str(self.id)

        self.fullPath        = env_folder + '/' + file_name
        self.fullPathWithExt = self.fullPath + '.txt'

    def run(self, agent, params):
        self.__makeFile(agent.env_name)

        print("Running experiment " + str(self.id))
        print("Params:", params)

        with open(self.fullPathWithExt, 'w') as f:
            f.write(str(params))
            f.write(agent.summary())

            agent.reset()
            #results, x_scale = agent.train(params)
            results, x_scale = agent.train2(params)
            
            DQN.plot(results, self.fullPath, self.id, [i * x_scale for i in range(len(results))])
            f.write("Final score: " + str(results[len(results)-1]) + '\n')
            print("Final score: " + str(results[len(results)-1]))
            print("Results stored at: " + self.fullPathWithExt)

if __name__ == "__main__":
    main()