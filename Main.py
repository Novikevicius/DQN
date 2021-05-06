from Agent import QT_Agent
import DQN

from functools import *
from datetime import datetime

import QTable

def main():
    qt_exp = Experiment(id=9999, agent_name='QT_Agent')
    qt_agent = QT_Agent('CartPole-v1', model=[QTable.Input(0, 15, 1, 0)])
    qt_exp.run(qt_agent, params={'epochs': 10000, 'lr': 0.5, 'gamma': 0.99})
    qt_exp.run(qt_agent, params={'epochs': 10000, 'lr': 0.1, 'gamma': 0.99})


class Experiment:
    experiments_folder = 'experiments'
    def __init__(self, id, agent_name) -> None:
        import os
        self.id = id 
        self.agent_name = agent_name
        self.folder = self.experiments_folder + '/' + self.agent_name   

        #create folders to store experiment's data
        if not os.path.exists(self.experiments_folder):
            os.mkdir(self.experiments_folder)
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def __makeFile(self):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        file_name       = dt_string + '_ID_' + str(self.id)

        self.fullPath        = self.folder + '/' + file_name
        self.fullPathWithExt = self.fullPath + '.txt'

    def run(self, agent, params):
        self.__makeFile()

        print("Running experiment " + str(self.id) + ":")
        print("Params:", params)

        with open(self.fullPathWithExt, 'w') as f:
            f.write(str(params))
            f.write(agent.summary())

            agent.reset()
            results, x_scale = agent.train(params)
            
            DQN.plot(results, self.fullPath, self.id, [i * x_scale for i in range(len(results))])
            f.write("Final score: " + str(results[len(results)-1]) + '\n')
            print("Final score: " + str(results[len(results)-1]))

if __name__ == "__main__":
    main()