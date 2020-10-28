import random
import gym
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

MODELS_FOLDER = './models/'

def createModel():
    global env
    model = Sequential()
    model.add(Dense(10, input_shape=env.observation_space.shape, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(env.action_space.n, activation='softmax' ))
    
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    return model

def train(ID, agent, epochs=1000, batchSize = 32):
    global env
    memory = deque(maxlen=200)
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
                    print("Episode: ", e, "Score", max_score)
                    max_score = 0
                break
        results.append(score)
        batch_size = batchSize
        if len(memory) < batch_size:
            batch_size = len(memory)
        data = random.sample(memory, batch_size)
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                target = r + gamma * np.max(agent.predict(s_new))
            T_s = agent.predict(np.array(s))
            T_s[0][a] = (target-T_s[0][a])**2
            agent.fit(s, T_s, epochs=1, verbose=0)
    agent.save(MODELS_FOLDER + str(ID))
    return results

def plot(results):
    plt.title("Score per episode")
    plt.plot(results)
    plt.show()

def test(ID, eps):
    global env
    from keras.models import load_model
    #agent = createModel()
    agent = load_model(MODELS_FOLDER + str(ID))
    max_score = 0
    for e in range(eps):
        state = np.array([env.reset()])
        done = False
        score = 0
        print("\t\Epoch: ", e)
        for i in range(300):
            action = np.argmax(agent.predict(state))
            new_state, reward, done, _ = env.step(action)
            new_state = np.array([new_state])
            env.render()
            state = new_state

            if done:
                break
            score += 1
        if score > max_score:
            max_score = score
        print(score)
    env.close()

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    lr = 0.01
    gamma = 0.95
    agent = createModel()
    agent.summary()
    r = None
    running = True
    ID = 0
    while running:
        command = input("Choose action: \n\t- q - quit \n\t- t - train \n\t- p - plot \n\t- c - create new agent \n\t- r - test agent \n\t- a - change agent \n")
        if command=='q':
            running = False
            break
        if command=='t':
            r = train(ID, agent, 100)
        if command == 'p' and r:
            plot(r)
        if command=='r':
            test(ID, 200)
        if command=='c':
            agent = createModel()
        if command=='a':
            temp = input("Choose ID: \n")
            ID = int(temp)
    env.close()