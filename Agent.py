from abc import *
import gym
import numpy as np
import random

class Agent(metaclass=ABCMeta):
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.createEnvironment()
        
        self.min_exploration = 0.01
        self.max_exploration = 1
        self.exploration_decay = 0.01
        self.lr = 0.1
        self.gamma = 0.99

    def createEnvironment(self):
        self.env = gym.make(self.env_name)

    def summary(self):
        return self.env_name
    
    @abstractmethod
    def train(self, params):
        pass

    @abstractclassmethod
    def action(self, state, epsilon):
        pass
    @abstractclassmethod
    def evaluate(self, state, action):
        pass
    @abstractclassmethod
    def updateValue(self, state, action, old_value, new_value):
        pass
    
    def train2(self, params):
        self.epsilon = 1
        epochs = params['epochs']
        result_x_size = round(epochs * 0.1) if 'x_size' not in params else params['x_size']
        result_x_size = 1 if result_x_size == 0 else result_x_size
        min_exploration_rate = self.min_exploration if 'min_expl' not in params else params['min_expl']
        max_exploration_rate = self.max_exploration if 'max_expl' not in params else params['max_expl']
        exploration_decay_rate = self.exploration_decay if 'expl_decay' not in params else params['expl_decay']
        max_steps = self.max_steps if 'max_steps' not in params else params['max_steps']
        lr = self.lr if 'lr' not in params else params['lr']
        gamma = self.gamma if 'gamma' not in params else params['gamma']
        sumup_reward_func = sum if 'reward_function' not in params else params['reward_function']
        #sumup_reward_func = max if 'reward_function' not in params else params['reward_function']
        epsilon = 1

        rewards = []

        for e in range(epochs):
            state = self.env.reset()
            done = False
            r = 0
            rewards_per_episode = []
            for s in range(max_steps):
                # choose an action for given state using epsilon-greedy approach
                action = self.action(state, epsilon)
                # perform chosen action and observ new state, given reward and if game finished
                new_state, reward, done, _ = self.env.step(action)
                # recalculate new Q-value using Bellman-Ford equation
                q_old = self.evaluate(state, action)
                q_new = q_old * (1-lr) + lr * (reward + gamma * np.max(self.evaluate(new_state)))
                # update Q-value
                self.updateValue(state, action, q_old, q_new)
                # move to the next environment state
                state = new_state
                # increase total reward for this episode
                rewards_per_episode.append(reward)
                r += reward

                if done:
                    break
                
            r = sumup_reward_func(rewards_per_episode)
            epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
            rewards.append(r)
        
        rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
        count = result_x_size

        results = [] # average rewards per result_x_size episodes
        for r in rewards_per_x_episodes:
            results.append(sum(r/result_x_size))
            count += result_x_size
        return results, result_x_size

    @abstractmethod
    def reset():
        pass

import QTable
class QT_Agent(Agent):
       
    def __init__(self, env_name, model=None) -> None:
        super().__init__(env_name)
        if model == None:
            self.model = [QTable.Input(-1, 1, 0.1, 4), 
                            QTable.Input(-1, 1, 0.1, 4),
                            QTable.Input(-1, 1, 0.1, 4),
                            QTable.Input(-1, 1, 0.1, 4)]
        else:
            self.model = model 
        self.createTable()
            
        #default values
        self.min_exploration = 0.01
        self.max_exploration = 1
        self.exploration_decay = 0.01
        self.max_steps = 500
        self.lr = 0.1
        self.gamma = 0.99

    def createTable(self):
        self.table = QTable.QTable(self.env.action_space.n, model=self.model)

    def action(self, state, epsilon=None):
        if epsilon == None or random.uniform(0, 1) > epsilon:
            action = np.argmax(self.table.getValue(state))
        else:
            action = self.env.action_space.sample()
        return action
    
    def evaluate(self, state, action=None):
        if action == None:
            return self.table.getValue(state)
        return self.table.getValue(state)[action]

    def updateValue(self, state, action, old_value, new_value):
        return self.table.setValue(state, action, new_value)

    def reset(self):
        self.table = QTable.QTable(self.env.action_space.n, model=self.model)

    def train(self, params):        
        # get required parameters
        epsilon = 1
        epochs = params['epochs']
        min_exploration_rate = self.min_exploration if 'min_expl' not in params else params['min_expl']
        max_exploration_rate = self.max_exploration if 'max_expl' not in params else params['max_expl']
        exploration_decay_rate = self.exploration_decay if 'expl_decay' not in params else params['expl_decay']
        max_steps = self.max_steps if 'max_steps' not in params else params['max_steps']
        lr = self.lr if 'lr' not in params else params['lr']
        gamma = self.gamma if 'gamma' not in params else params['gamma']

        rewards = []
        result_x_size = round(epochs * 0.1)
        result_x_size = 1 if result_x_size == 0 else result_x_size


        for e in range(epochs):
            state = self.env.reset()
            done = False
            r = 0
            for s in range(max_steps):
                action = self.action(state, epsilon)
                new_state, reward, done, _ = self.env.step(action)
                q_new = self.table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(self.table.getValue(new_state)))
                self.table.setValue(state, action, q_new)
                #table.setValue(state, action, q_new, e < 100)
                state = new_state
                r += reward
                if done:
                    break
            epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
            rewards.append(r)
        
        rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
        count = result_x_size

        results = [] # average rewards per result_x_size episodes
        for r in rewards_per_x_episodes:
            results.append(sum(r/result_x_size))
            count += result_x_size
        return results, result_x_size
        