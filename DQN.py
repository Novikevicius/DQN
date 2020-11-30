from QTable import QTable
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

MODELS_FOLDER = 'experiments/DQN_Agent/models/'

class DQN_Agent():
    def __init__(self, env, ID, lr=0.001, activation_fn='linear', loss_fn='mse', filename=None , use_target_network=True):
        self.env = env
        self.ID = ID
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.use_target_network = use_target_network
        self.memory = deque(maxlen=500)
        if filename:
            from keras.models import load_model
            self.agent = load_model(filename)
        else:
            self.agent = Sequential()
            self.agent.add(Dense(25, input_shape=env.observation_space.shape, activation='relu'))
            self.agent.add(Dense(25, activation='relu'))
            self.agent.add(Dense(env.action_space.n, activation=activation_fn ))        
            self.agent.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
            
            if self.use_target_network:
                self.target = keras.models.clone_model(self.agent)

    def train(self, gamma=0.99, epochs=1000, batchSize = 50, file=None):
        results = []
        max_score = 0
        for e in range(epochs):
            done = False
            score = 0
            state = np.array([self.env.reset()])
            for i in range(500):
                action = np.argmax(self.agent.predict(state))
                new_state, reward, done, _ = self.env.step(action)
                new_state = np.array([new_state])

                self.rember([state, action, reward, new_state, done])

                score += 1
                state = new_state

                if done:
                    if score > max_score:
                        max_score = score
                    if e % 10 == 0:
                        log_msg = "Epoch: " +  str(e) + ", score" +  str(max_score)
                        if file:
                            file.write(log_msg + '\n')
                        else:
                            print(log_msg)
                        max_score = 0
                    break
            results.append(score)
            self.replay(batchSize)
            if self.use_target_network and e % 50 == 0 and e != 0:
                self.target = keras.models.clone_model(self.agent)

        self.save()
        return results
    def rember(self, s):
        self.memory.append(s)
    def replay(self, gamma=0.99, batch_size=50):
        if len(self.memory) < batch_size:
            return
        data = random.sample(self.memory, batch_size)
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                if self.use_target_network:
                    prediction = self.target.predict(s_new)
                else:
                    prediction = self.agent.predict(s_new)
                target = r + gamma * np.max(prediction[0])
            T_s = self.agent.predict(np.array(s))
            T_s[0][a] = target
            self.agent.fit(s, T_s, verbose=0)

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

def plot(results, saveFolder=None, ID=0):
    plt.title("Score per episode")
    plt.plot(results)
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
            run_DQN_experiments()
        if command=='l':
            import tkinter as tk
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(initialdir=MODELS_FOLDER, title="Pasirinkite agento modeli")
            agent = DQN_Agent(env, ID=999, filename=file_path)
def run_DQN_experiments():
    global env
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQN_Agent/models/'
    experiment_ID_file = 'experiments/DQN_Agent/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
    ID = run_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()
    
def run_q_table_experiments():
    global env
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/models/'
    experiment_ID_file = 'experiments/Q_Table/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
    #ID = run_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse')
    ID = run_q_table_experiment(ID, 100, 0.01)
   # with open(experiment_ID_file, 'w') as f:
   #     f.write(str(ID)+'\n')
   #     f.close()

def run_q_table_experiment(ID, epochs=100, lr=0.01):
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'Q_Table'
    folder          = experiments_folder + '/' + agent_folder
    file            = str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    # train agent
    table = QTable(10, 3)
    table.split(0, 50)
    print(table)
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)     + ':\n')
        f.write("Epochs: "        + str(epochs) + '\n')
        f.write("Learning rate: " + str(lr)     + '\n')

        #plot(r, fullPath, ID)
        #f.write("Final score: " + str(r[len(r)-1]) + '\n')
        #print("Final score: " + str(r[len(r)-1]))
    return ID+1

def run_experiment(ID, epochs = 100, lr=0.01, gamma=0.99, activation='linear', loss='mse'):
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'DQN_Agent'
    folder          = experiments_folder + '/' + agent_folder
    file            = str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    agent = DQN_Agent(env, ID, lr, activation, loss)
    print("Running experiment " + str(ID) + ":")
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
    import sys
    global env
    env = gym.make("CartPole-v1")

    #run()
    #run_DQN_experiments()
    run_q_table_experiments()

    env.close()
    sys.exit(0)