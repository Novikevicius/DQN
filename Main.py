from collections import deque
from Agent import QT_Agent
import DQN
from QTable import *

from functools import *
from datetime import datetime
import os
import random

import gym

def main():
    model = [Input(-1, 1, 0.1, 4, static=False), 
             Input(-1, 1, 0.1, 4, static=False),
             Input(-1, 1, 0.1, 4, static=False),
             Input(-1, 1, 0.1, 4, static=False)]
    import DQTable as dqt
    #table = dqt.DQTable(2, model=model)

    table = QTable(2, model=model, dynamic=True)
    env = gym.make("CartPole-v0")
    epochs = 100000
    min_exploration_rate = 0.1
    max_exploration_rate = 1
    exploration_decay_rate = 0.01
    epsilon = 1

    result_x_size = 1000

    lr = 0.05
    gamma = 0.99

    rewards = []
    for e in range(epochs):
        state = env.reset()
        for s in range(500):
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(table.getValue(state))
            else:
                action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            
            q_old = table.getValue(state)[action]
            q_new = q_old * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)

            state = new_state
            if done:
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        print("E:", e, "score:", s, "epsilon:", epsilon)
        rewards.append(s)

    count = 0
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    xs =[i * result_x_size for i in range(len(results))]
    DQN.plot(results, 'DQTable_12', 12,xs=xs)
    max_r = np.argmax(results)
    print(table, '\n')
    print(table.get_intervals(), '\n')
    print("Max", results[max_r], "after", max_r+1, "epochs")
    print("Final", results[len(results)-1])
    print("Table size", table.n)
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