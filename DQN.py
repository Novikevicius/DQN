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

MODELS_FOLDER = './models/'

def createModel(lr=0.001, activation_fn='linear', loss_fn='mse'):
    global env
    model = Sequential()
    model.add(Dense(25, input_shape=env.observation_space.shape, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(env.action_space.n, activation=activation_fn ))
    
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    return model

def train(ID, agent, gamma=0.99, epochs=1000, batchSize = 50):
    global env
    memory = deque(maxlen=500)
    results = []
    max_score = 0
    for e in range(epochs):
        state = np.array([env.reset()])
        done = False
        score = 0
        for i in range(250):
            action = np.argmax(agent.predict(state))

            new_state, reward, done, _ = env.step(action)

            new_state = np.array([new_state])

            memory.append([state, action, reward, new_state, done])
            score += 1
            state = new_state

            if done:
                if score > max_score:
                    max_score = score
                if e % 50 == 0:
                    print("Epoch: ", e, "Score", max_score)
                    max_score = 0
                break
        results.append(score)
        batch_size = batchSize
        if len(memory) < batch_size:
            continue
        data = random.sample(memory, batch_size)
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                target = r + gamma * np.max(agent.predict(s_new)[0])
            T_s = agent.predict(np.array(s))
            T_s[0][a] = target
            agent.fit(s, T_s, verbose=0)
    agent.save(MODELS_FOLDER + str(ID))
    return results

def plot(results, saveFolder=None, ID=0):
    plt.title("Score per episode")
    plt.plot(results)
    if saveFolder:
        plt.savefig(saveFolder)
        plt.clf()
    else:
        plt.show()

def test(ID, eps, render=True):
    global env
    from keras.models import load_model
    #agent = createModel()
    agent = load_model(MODELS_FOLDER + str(ID))
    max_score = 0
    for e in range(eps):
        state = np.array([env.reset()])
        done = False
        score = 0
        for i in range(300):
            action = np.argmax(agent.predict(state))
            new_state, reward, done, _ = env.step(action)
            new_state = np.array([new_state])
            if render:
                env.render()
            state = new_state

            if done:
                break
            score += 1
        if score > max_score:
            max_score = score
        print("Epoch:", e, "score", score)
    print("Max score", max_score)

def run():
    global env
    lr = 0.001
    gamma = 0.99
    ID = 1
    agent = createModel()
    agent.summary()
    r = []
    running = True
    while running:
        command = input("Agent: " + str(ID) + "\nChoose action: \n\t- q - quit \n\t- t - train \n\t- p - plot \n\t- c - create new agent \n\t- r - test agent \n\t- a - change agent \n")
        if command=='q':
            running = False
            break
        if command=='t':
            r.extend(train(ID, agent, 100))
        if command == 'p' and r:
            plot(r)
        if command=='r':
            test(ID, 5)
        if command=='c':
            agent = createModel()
            r = []
        if command=='a':
            temp = input("Choose ID: \n")
            ID = int(temp)
            r = []
        if command=='s':
            plot(r, MODELS_FOLDER + str(ID), ID)
        if command=='l':
            agent = load_model(MODELS_FOLDER + str(ID))

def run_experiment(ID, epochs = 100, lr=0.01, gamma=0.99, activation='linear', loss='mse'):
    global env
    folder          = 'experiments'
    file            = str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(folder):
        os.mkdir(folder)

    agent = createModel(lr, activation, loss)
    print("Running experiment " + str(ID) + ":")
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)     + ':\n')
        f.write("Epochs: "        + str(epochs) + '\n')
        f.write("Learning rate: " + str(lr)     + '\n')
        f.write("Gamma: "         + str(gamma)  + '\n')
        f.write("Activation: "    + activation  + '\n')
        f.write("Loss: "          + loss        + '\n\n')
        f.write("Model summary:\n")
        agent.summary(print_fn=lambda s: f.write(s + '\n'))

        r = train(ID, agent,gamma=gamma,  epochs=epochs, batchSize=100)
        plot(r, fullPath, ID)

    return ID+1

if __name__ == "__main__":
    import sys
    
    experiment_ID_file = 'exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    env = gym.make("CartPole-v1")
    ID = run_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    ID = run_experiment(ID, 500, 0.1,   0.99, 'linear', 'mse')
    ID = run_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
    env.close()

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()
    sys.exit(0)