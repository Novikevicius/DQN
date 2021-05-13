from os import system
import random
import gym
from keras.engine.saving import load_model
import matplotlib.pyplot as plt
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.image as mpimg
from IPython import display
import os
import QTable
import sys
import Agent
MODELS_FOLDER = 'experiments/DQN_Agent/models/'
CARTPOLE_ENV_NAME = "CartPole-v0"
FROZENLAKE_ENV_NAME = "FrozenLake-v0"

def start():
    #DQT experiments
    run_DQT_cartpole_experiments()

    #DQN experiments
    #run_DQN_frozenlake_experiments()
    #run_DQN_cartpole_experiments()

    #QT experiments
    #run_QT_frozen_lake_experiments()
    #run_QT_cartpole_experiments()
    pass

class DQN_Agent(Agent.Agent):
    def __init__(self, env_name, ID, lr=0.001, activation_fn='linear', loss_fn='mse', filename=None , use_target_network=True, params=None):
        super().__init__(env_name)
        self.ID = ID
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.use_target_network = use_target_network
        self.memory = deque(maxlen=100000)
        if filename:
            from keras.models import load_model
            self.agent = load_model(filename)
        else:
            self.params = {'activation':activation_fn, 'loss':loss_fn, 'lr':lr} if params == None else params
            self.agent = self.createModel()
            
            if self.use_target_network:
                self.target = keras.models.clone_model(self.agent)

                '''
        #default values
        self.min_exploration = 0.01
        self.max_exploration = 1
        self.exploration_decay = 0.01
        self.max_steps = 500
        self.lr = 0.001
        self.gamma = 0.99
        self.activation_fn = 'linear'
        self.loss = 'mse'
        '''
        self.min_exploration = 0.01
        self.max_exploration = 1
        self.exploration_decay = 0.01

    def summary(self):
        from io import StringIO
        stream = StringIO()
        self.agent.summary(print_fn=lambda s: stream.write(s + '\n'))
        model_summary = stream.getvalue()
        stream.close()
        return super().summary() + ':\n ' + model_summary

    def createTable(self):
        self.table = QTable.QTable(self.env.action_space.n, model=self.model)
    
    def action(self, state, epsilon=None):
        state = np.array([state])
        if epsilon == None or random.uniform(0, 1) > epsilon:
            action = np.argmax(self.agent.predict(state)[0])
        else:
            action = self.env.action_space.sample()
        return action
        
    def evaluate(self, state, action=None):
        state = np.array([state])
        if action == None:
            return self.agent.predict(state)[0]
        return self.agent.predict(state)[0][action]
        
    def updateValue(self, state, action, reward, new_state, done, old_value, new_value):
        self.memory.append([state, action, reward, new_state, done])
        return
        state = np.array([state])
        s = self.agent.predict(np.array(state))
        s[0][action] = new_value
        self.agent.fit(state, s, verbose=0)

    def createModel(self):
        params = self.params
        lr = self.lr if 'lr' not in params else params['lr']
        activation_fn = self.activation_fn if 'activation' not in params else params['activation']
        loss = self.loss if 'loss' not in params else params['loss']

        agent = Sequential()
        agent.add(Dense(25, input_shape=self.env.observation_space.shape, activation='relu'))
        agent.add(Dense(50, activation='relu'))
        agent.add(Dense(100, activation='relu'))
        agent.add(Dense(50, activation='relu'))
        agent.add(Dense(25, activation='relu'))
        agent.add(Dense(self.env.action_space.n, activation=activation_fn ))        
        agent.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
        return agent
    
    def reset(self):
        self.createModel()

    def onEpisodeEnd(self):
        self.replay(self.gamma)
        return
        gamma = self.gamma
        batch_size = 50
        if len(self.memory) < batch_size:
            return
        data = random.sample(self.memory, batch_size)
        for s, a, new_value in data:
            state = np.array([s])   
            st = self.agent.predict(np.array(state))
            st[0][a] = new_value
            self.agent.fit(state, st, verbose=0)
        
    def train(self, gamma=0.99, epochs=1000, batchSize = 50, file=None, params=None):
        # get required parameters
        if params:
            epsilon = 1
            epochs = params['epochs']
            min_exploration_rate = self.min_exploration if 'min_expl' not in params else params['min_expl']
            max_exploration_rate = self.max_exploration if 'max_expl' not in params else params['max_expl']
            exploration_decay_rate = self.exploration_decay if 'expl_decay' not in params else params['expl_decay']
            max_steps = self.max_steps if 'max_steps' not in params else params['max_steps']
        self.epsilon = 1
        min_exploration_rate = self.min_exploration
        max_exploration_rate = self.max_exploration
        exploration_decay_rate = self.exploration_decay
        results = []
        for e in range(epochs):
            done = False
            score = 0            
            state = np.array([self.env.reset()])

            for i in range(500):                
                if self.epsilon == None or random.uniform(0, 1) > self.epsilon:
                    #action = np.argmax(self.table.getValue(state))
                    action = np.argmax(self.agent.predict(state))
                else:
                    action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)
                new_state = np.array([new_state])

                self.rember([state, action, reward, new_state, done])

                score += 1
                state = new_state

                if done:
                    if e % 10 == 0:
                        log_msg = "Epoch: " +  str(e) + ", score"# +  str(max_score)
                        if file:
                            file.write(log_msg + '\n')
                        else:
                            print(log_msg)
                    break
            results.append(score)
            self.replay(batchSize)
            if self.use_target_network and e % 50 == 0 and e != 0:
                self.target = keras.models.clone_model(self.agent)
                
            self.epsilon = self.min_exploration + (self.max_exploration - self.min_exploration) * np.exp(-self.exploration_decay*e)
        print("E:", e, "score:", i, "epsilon:", epsilon)

        self.save()
        return results, 1
    def rember(self, s):
        self.memory.append(s)
    def replay(self, gamma=0.99, batch_size=256):

        data = random.sample(self.memory, min(len(self.memory), batch_size))
        xs = []
        ys = []
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                if self.use_target_network:
                    prediction = self.target.predict(np.array([s_new]))
                else:
                    prediction = self.agent.predict(np.array([s_new]))
                target = r + gamma * np.max(prediction[0])
            T_s = self.agent.predict(np.array([s]))
            T_s[0][a] = target
            xs.append(s)
            ys.append(T_s[0])
        self.agent.fit(np.array(xs), np.array(ys), batch_size=len(xs), verbose=0)

    def save(self, filename=None):
        if filename == None:
            filename = MODELS_FOLDER + str(self.ID)
        self.agent.save(filename)

    def test(self, epochs=10, render=True):
        max_score = 0
        for e in range(epochs):
            state = np.array([self.env.reset()])
            done = False
            score = 0
            for i in range(300):
                action = np.argmax(self.agent.predict(state))
                new_state, reward, done, _ = self.env.step(action)
                new_state = np.array([new_state])
                if render:
                    self.env.render()
                state = new_state

                if done:
                    break
                score += 1
            if score > max_score:
                max_score = score
            print("Epoch:", e, "score", score)
        print("Max score", max_score)

def plot(results, saveFolder=None, ID=0, xs=None, x_size=None, max_score=None, max_score_after=None, lr=None):
    if x_size == 1:
        plt.title("Surenkamų taškų skaičius per epochą")
    else:
        plt.title("Vidutinis surenkamų taškų skaičius" if x_size == None else "Paskutinių {0} epochų taškų skaičius".format(x_size))
    plt.xlabel('Epochų skaičius')
    plt.ylabel('Taškai')
    if not xs:
        plt.plot(results)
    else:
        plt.plot(xs, results)
    if not (max_score == None):
        plt.axvline(max_score_after, ymax=max_score, color="r", linestyle="--")
        plt.legend(["Vidutinis taškų skaičius" if lr == None else 'Mokymo greitis {0}'.format(lr) , "{0} taškų skaičius po {1} epochų".format(max_score, max_score_after)], loc ="lower right")
    if saveFolder:
        plt.savefig(saveFolder)
    else:
        plt.show()
    plt.clf()

def run():
    global env
    lr = 0.001
    gamma = 0.99
    ID = 1
    agent = DQN_Agent(env, ID, lr, 'linear')
    agent.agent.summary()
    r = []
    running = True
    while running:
        command = input("Agent: " + str(ID) + "\nChoose action: \n\t- e - run DQN experiments \n\t- q - quit \n\t- t - train \n\t- p - plot \n\t- c - create new agent \n\t- r - test agent \n\t- a - change agent \n")
        if command=='q':
            running = False
            break
        if command=='t':
            r.extend(agent.train(gamma, 100))
        if command == 'p' and r:
            plot(r)
        if command=='r':
            agent.test(5)
        if command=='c':
            agent = DQN_Agent(env, ID, lr, 'linear')
            r = []
        if command=='a':
            temp = input("Choose ID: \n")
            ID = int(temp)
            r = []
        if command=='s':
            plot(r, MODELS_FOLDER + str(ID), ID)
        if command=='e':
            run_DQN_cartpole_experiments()
        if command=='l':
            import tkinter as tk
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(initialdir=MODELS_FOLDER, title="Pasirinkite agento modeli")
            agent = DQN_Agent(env, ID=999, filename=file_path)

def run_DQN_cartpole_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQN_Agent/{0}/models/'.format(CARTPOLE_ENV_NAME)
    experiment_ID_file = 'experiments/DQN_Agent/{0}/exp_ID.txt'.format(CARTPOLE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_DQN_CartPole_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    ID = run_DQN_CartPole_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
    ID = run_DQN_CartPole_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_DQN_frozenlake_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQN_Agent/{0}/models/'.format(FROZENLAKE_ENV_NAME)
    experiment_ID_file = 'experiments/DQN_Agent/{0}/exp_ID.txt'.format(FROZENLAKE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_DQN_CartPole_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    ID = run_DQN_FrozenLake_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
    ID = run_DQN_FrozenLake_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_DQN_FrozenLake_experiment(ID, epochs=500, lr=0.001, gamma=0.90, activation='linear', loss='mse'):
    global env
    env = gym.make(FROZENLAKE_ENV_NAME)
    run_experiment(ID, epochs, lr, gamma, activation, loss, FROZENLAKE_ENV_NAME)

def run_DQN_CartPole_experiment(ID, epochs=500, lr=0.001, gamma=0.90, activation='linear', loss='mse'):
    global env
    env = gym.make(CARTPOLE_ENV_NAME)
    run_experiment(ID, epochs, lr, gamma, activation, loss, CARTPOLE_ENV_NAME)

def run_QT_frozen_lake_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/{0}/models/'.format(FROZENLAKE_ENV_NAME)
    experiment_ID_file = 'experiments/Q_Table/{0}/exp_ID.txt'.format(FROZENLAKE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env    
    env = gym.make(FROZENLAKE_ENV_NAME)
    ID = run_QT_frozen_lake_experiment(ID, epochs=10000, lr=0.01, gamma=0.99, result_x_size=1000)
    ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_QT_cartpole_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/{0}/models/'.format(CARTPOLE_ENV_NAME)
    experiment_ID_file = 'experiments/Q_Table/{0}/exp_ID.txt'.format(CARTPOLE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env
    env = gym.make(CARTPOLE_ENV_NAME)
    ID = run_QT_cartpole_experiment(ID, epochs=100000, lr=0.5,  gamma=0.99, result_x_size=1000)
    ID = run_QT_cartpole_experiment(ID, epochs=100000, lr=0.7,  gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_DQT_cartpole_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQT/{0}/models/'.format(CARTPOLE_ENV_NAME)
    experiment_ID_file = 'experiments/DQT/{0}/exp_ID.txt'.format(CARTPOLE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env
    env = gym.make(CARTPOLE_ENV_NAME)
    model = [QTable.Input(-0.1, 0.1, 0.1, 4, static=False),
            QTable.Input(-0.1, 0.1, 0.1, 4, static=False),
            QTable.Input(-0.1, 0.1, 0.1, 4, static=False),
            QTable.Input(-0.1, 0.1, 0.1, 4, static=False),]
    ID = run_dqt_cartpole_experiment(ID, epochs=100000, lr=0.1,  gamma=0.99, result_x_size=100)
    ID = run_dqt_cartpole_experiment(ID, epochs=100000, lr=0.5,  gamma=0.99, result_x_size=100)
    ID = run_dqt_cartpole_experiment(ID, epochs=100000, lr=0.7,  gamma=0.99, result_x_size=100)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_QT_frozen_lake_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100):
    global env
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    experiments_folder = 'experiments'
    agent_folder    = 'Q_Table/{0}'.format(FROZENLAKE_ENV_NAME)
    folder          = experiments_folder + '/' + agent_folder
    file            = '{0}_{1}'.format(FROZENLAKE_ENV_NAME, ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    model = [QTable.Input(0, 15, 1, 0)]
    table = QTable.QTable(env.action_space.n, model=model)
    rewards = []
    max_steps = 200
    max_score = 0
    max_score_after = 0
    for e in range(epochs):
        state = env.reset()
        done = False
        r = 0
        for s in range(max_steps):
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            v = table.getValue(state)
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 100)
            state = new_state
            r += reward
            if done:
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
        if max_score < r:
            max_score = r
            max_score_after = e
        print("E:", e, "score:", s, "epsilon:", epsilon)
    
    max_score = None if max_score == 0 else max_score
    max_score_after = None if max_score == None else max_score_after
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)        + ':\n')
        f.write("Epochs: "        + str(epochs)    + '\n')
        f.write("Learning rate: " + str(lr)        + '\n')
        f.write("Gamma: "         + str(gamma)     + '\n')
        f.write("Last Reward: "   + str(rewards[len(rewards)-1])     + '\n')

        plot(results, fullPath, ID, xs=[i for i in range(result_x_size, epochs+1, result_x_size)], x_size=result_x_size, max_score=max_score, max_score_after=max_score_after, lr=lr)
        f.write("Max score: " + str(max_score) + " after " + str(max_score_after) + '\n')
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(results[len(results)-1]))
        print("Max score: " + str(max_score) + " after " + str(max_score_after) + '\n')
    table.save(MODELS_FOLDER+str(ID))
    return ID+1

def run_QT_cartpole_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100):
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'Q_Table/{0}'.format(CARTPOLE_ENV_NAME)
    folder          = experiments_folder + '/' + agent_folder
    file            = '{0}_{1}'.format(CARTPOLE_ENV_NAME, ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.01
    model = [QTable.Input(-1, 1, 0.1, 4, static=True), 
             QTable.Input(-1, 1, 0.1, 4, static=True),
             QTable.Input(-1, 1, 0.1, 4, static=True),
             QTable.Input(-1, 1, 0.1, 4, static=True)]
    import DQTable as dqt
    table = dqt.DQTable(env.action_space.n, model=model)
    rewards = []
    max_steps = 500
    for e in range(epochs):
        state = env.reset()
        done = False
        r = 0
        for s in range(max_steps):
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 100)
            state = new_state
            #r += reward
            if done:
                r = s
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
        print("E:", e, "score:", s, "epsilon:", epsilon)
    
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)        + ':\n')
        f.write("Epochs: "        + str(epochs)    + '\n')
        f.write("Learning rate: " + str(lr)        + '\n')
        f.write("Gamma: "         + str(gamma)     + '\n')
        f.write("Last Reward: "   + str(rewards[len(rewards)-1])     + '\n')

        plot(results, fullPath, ID, xs=[i for i in range(result_x_size, epochs+1, result_x_size)])
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(results[len(results)-1]))
    table.save(MODELS_FOLDER+str(ID))
    return ID+1

def run_dqt_cartpole_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100, model=None):
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'DQT/{0}'.format(CARTPOLE_ENV_NAME)
    folder          = experiments_folder + '/' + agent_folder
    file            = '{0}_'.format(CARTPOLE_ENV_NAME)+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running DQT CartPole experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.01
    if model == None:
        model = [QTable.Input(-1, 1, 0.1, 4, static=False), 
                QTable.Input(-1, 1, 0.1, 4, static=False),
                QTable.Input(-1, 1, 0.1, 4, static=False),
                QTable.Input(-1, 1, 0.1, 4, static=False)]
    table = QTable.QTable(env.action_space.n, model=model, dynamic=True)
    rewards = []
    max_steps = 500
    max_score = 0
    max_score_after = 0
    for e in range(epochs):
        state = env.reset()
        done = False
        r = 0
        for s in range(max_steps):
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 100)
            state = new_state
            #r += reward
            if done:
                r = s
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
        if max_score < r:
            max_score = r
            max_score_after = e
        print("E:", e, "score:", s, "epsilon:", epsilon)
    
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)        + ':\n')
        f.write("Epochs: "        + str(epochs)    + '\n')
        f.write("Learning rate: " + str(lr)        + '\n')
        f.write("Gamma: "         + str(gamma)     + '\n')
        f.write("Max Reward: "   + str(rewards[len(rewards)-1])     + '\n')
        f.write("Last Reward: "   + str(rewards[len(rewards)-1])     + '\n')

        plot(results, fullPath, ID, xs=[i for i in range(result_x_size, epochs+1, result_x_size)], x_size=result_x_size, max_score=max_score, max_score_after=max_score_after,lr=lr)
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        f.write("Starting intervals: " + str(QTable.QTable(env.action_space.n, model=model, dynamic=True).get_intervals()) + '\n')

        f.write("Intervals: " + str(table.get_intervals()) + '\n')
        print("Final score: " + str(results[len(results)-1]))
        print("Max score: " + str(max_score) + " after " + str(max_score_after) + '\n')
        print("Results saved to", fullPathWithExt)
    table.save(MODELS_FOLDER+str(ID))
    return ID+1

def load_frozen_lake_agent(ID):
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/FrozenLake/models/'
    return QTable.QTable.load(MODELS_FOLDER+str(ID))

def play_frozen_lake(table, n=1, verbose=0):
    # verbose 0 - no priniting, with visualization, 1 - only printing, 2 - print and plot graph, 3 - only plot graph
    env = gym.make(FROZENLAKE_ENV_NAME)
    epsilon = 1
    rewards = []
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    for e in range(n):
        import time
        state = env.reset()
        done = False
        r = 0
        if verbose == 1 or verbose == 2:
            print("*****EPISODE ", e+1, "*****\n\n\n\n")
            time.sleep(1)
        for s in range(100):
            if verbose == 0:
                env.render()
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            r += reward
            
            if done:
                if verbose == 1 or verbose == 2:
                    print("Your reward:", reward)
                    time.sleep(3)
                break
        rewards.append(r)
    env.close()
    if verbose == 3 or verbose == 2:
        plot(rewards)
    return rewards
    
def run_experiment(ID, epochs = 100, lr=0.01, gamma=0.99, activation='linear', loss='mse', env_name=None):
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'DQN_Agent/{0}'.format(env_name)
    folder          = experiments_folder + '/' + agent_folder
    file            = '{0}_'.format(env_name)+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    agent = DQN_Agent(env, ID, lr, activation, loss)
    print("Running experiment {0} ID:{1}:".format(env_name, ID))
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)     + ':\n')
        f.write("Epochs: "        + str(epochs) + '\n')
        f.write("Learning rate: " + str(lr)     + '\n')
        f.write("Gamma: "         + str(gamma)  + '\n')
        f.write("Activation: "    + activation  + '\n')
        f.write("Loss: "          + loss        + '\n\n')
        f.write("Model summary:\n")
        agent.agent.summary(print_fn=lambda s: f.write(s + '\n'))

        r = agent.train(gamma=gamma,  epochs=epochs, batchSize=100, file=f)
        plot(r, fullPath, ID)
        f.write("Final score: " + str(r[len(r)-1]) + '\n')
        print("Final score: " + str(r[len(r)-1]))

    return ID+1

if __name__ == "__main__":
    
    start()
    sys.exit(0)
    global env
    env = gym.make("CartPole-v1")
    #env = gym.make("FrozenLake-v0")
   
    '''
    model = [QTable.Input(0, 1, 1, 0), QTable.Input(1, 2, 1, 0)]
    import DQTable as dqt
    table = dqt.DQTable(4, model=model)
    for a in range(4):
        table.setValue([-1, 0], a, (a+1))
    for a in range(4):
        table.setValue([1, 0], a, 2*(a+1))
    for a in range(4):
        table.setValue([2, 0], a, 3*(a+1))
        
    table.getValue([1, 0], True)
    for e in range(20):
        state = env.reset()
        for s in range(100):
            action = env.action_space.sample()
            table.getValue([2, 0], True)
            state, r, done, info = env.step(action)
    print(table)'''
    #agent = load_frozen_lake_agent(16)
    #play_frozen_lake(agent, n=1000, verbose=3)
    #run()
    #run_DQN_experiments()
    #run_QT_frozen_lake_experiments()
    #run_QT_cartpole_experiments()
    #run_QT_cartpole_experiments()

    env.close()
    sys.exit(0)