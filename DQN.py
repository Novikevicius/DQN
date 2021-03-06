import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense

import os
import QTable

MODELS_FOLDER = 'experiments/DQN_Agent/models/'
CARTPOLE_ENV_NAME = "CartPole-v0"
FROZENLAKE_ENV_NAME = "FrozenLake-v0"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def start():
    #DQT experiments
    run_DQT_frozenlake_experiments()
    run_DQT_cartpole_experiments()

    #DQN experiments
    run_DQN_frozenlake_experiments()
    run_DQN_cartpole_experiments()

    #QT experiments
    run_QT_frozen_lake_experiments()
    run_QT_cartpole_experiments()
    pass

class DQN_Agent():
    def __init__(self, env, ID, lr=0.001, activation_fn='linear', loss_fn='mse', filename=None , use_target_network=False, params=None, observation_space=None):
        self.env = env
        self.ID = ID
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.use_target_network = use_target_network
        self.memory = deque(maxlen=2000)
        self.observation_space = env.observation_space.shape if len(env.observation_space.shape) > 0 else env.observation_space.n
        self.frozen_lake = True
        if filename:
            from keras.models import load_model
            self.agent = load_model(filename)
        else:
            self.params = {'activation':activation_fn, 'loss':loss_fn, 'lr':lr} if params == None else params
            self.agent = self.createModel()
            
            if self.use_target_network:
                self.target = keras.models.clone_model(self.agent)
                self.target.build(self.observation_space)
                self.target.compile(loss=self.loss_fn, optimizer=keras.optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])
                self.target.set_weights(self.agent.get_weights())

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
    
    def action(self, state, epsilon=None):
        state = np.array([state])
        if epsilon == None or random.uniform(0, 1) > epsilon:
            action = np.argmax(self.agent.predict(state)[0])
        else:
            action = self.env.action_space.sample()
        return action

    def createModel(self):
        params = self.params
        lr = self.lr if 'lr' not in params else params['lr']
        activation_fn = self.activation_fn if 'activation' not in params else params['activation']
        loss = self.loss if 'loss' not in params else params['loss']

        agent = Sequential()
        if type(self.observation_space) is int:
            agent.add(Dense(24, input_dim=self.observation_space, activation='relu'))
        else:
            agent.add(Dense(24, input_shape=self.observation_space, activation='relu'))
        #agent.add(BatchNormalization(center=False, trainable=False))
        agent.add(Dense(24, activation='relu'))
        agent.add(Dense(self.env.action_space.n, activation=activation_fn ))        
        agent.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
        return agent

    def update_agent(self):
        self.agent = keras.models.clone_model(self.target)
        self.agent.build(self.observation_space)
        self.agent.compile(loss=self.loss_fn, optimizer=keras.optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])
        self.agent.set_weights(self.target.get_weights())
    def reset(self):
        self.createModel()
        
    def train(self, gamma=0.99, epochs=1000, batchSize = 50, file=None, params=None):
        # get required parameters
        if params:
            epochs = params['epochs']
        self.epsilon = 1
        def one_hot_encode(state):
            s = np.zeros(16)
            s[state] = 1
            return np.array([s])
        results = []
        for e in range(epochs):
            done = False
            score = 0            
            state = self.env.reset()
            if self.frozen_lake:
                state = one_hot_encode(state)
            else:
                state = np.reshape(state, [1,4])

            for i in range(500):                
                if self.epsilon == None or random.uniform(0, 1) > self.epsilon:
                    #action = np.argmax(self.table.getValue(state))
                    action = np.argmax(self.agent.predict(state)[0])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, _ = self.env.step(action)

                if self.frozen_lake:
                    new_state = one_hot_encode(new_state)
                    if done and reward == 0:
                        reward = -1
                else:
                    reward = reward if not done else -10
                    state = np.reshape(state, [1,4])
                    new_state = np.reshape(new_state, [1,4])

                self.remember([state, action, reward, new_state, done])
                state = new_state

                if done:
                    if self.frozen_lake:
                        reward = 0 if reward < 0 else reward
                    break
            score = reward if self.frozen_lake else i
            results.append(score)
            print("E:", e, "score:", i, "epsilon:", self.epsilon, "reward", reward)
            self.replay(gamma=gamma, batch_size=batchSize)
            if self.use_target_network and e % 5 == 0 and e != 0:
                self.update_agent()
            self.epsilon = self.min_exploration + (self.max_exploration - self.min_exploration) * np.exp(-self.exploration_decay*e)

        self.save()
        return results, 1
    def remember(self, s):
        self.memory.append(s)
    def replay(self, gamma=0.99, batch_size=256):

        data = random.sample(self.memory, min(len(self.memory), batch_size))
        xs = []
        ys = []
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                if self.use_target_network:
                    prediction = self.target.predict(s_new)[0]
                else:
                    prediction = self.agent.predict(s_new)[0]
                target = r + gamma * np.max(prediction)
            T_s = self.agent.predict(s)
            T_s[0][a] = target
            #xs.append(s[0])
            #ys.append(T_s[0])
            self.agent.fit(s, T_s, epochs=1, verbose=0)
        #self.agent.fit(np.array(xs), np.array(ys), epochs=1, batch_size=len(xs), verbose=0)

    def save(self, filename=None):
        if filename == None:
            filename = MODELS_FOLDER + str(self.ID)
        self.agent.save(filename)

def plot(results, saveFolder=None, ID=0, xs=None, x_size=None, max_score=None, max_score_after=None, lr=None):
    if x_size == 1:
        plt.title("Surenkam?? ta??k?? skai??ius per epoch??")
    else:
        plt.title("Vidutinis surenkam?? ta??k?? skai??ius" if x_size == None else "Paskutini?? {0} epoch?? ta??k?? vidurkis".format(x_size))
    plt.xlabel('Epoch?? skai??ius')
    plt.ylabel('Ta??kai')
    if not xs:
        plt.plot(results)
    else:
        plt.plot(xs, results)
    if not (max_score == None):
        plt.axvline(max_score_after, ymax=max_score, color="r", linestyle="--")
        plt.legend(["Vidutinis ta??k?? skai??ius" if lr == None else 'Mokymo greitis {0}'.format(lr) , "{0} ta??k?? skai??ius po {1} epoch??".format(max_score, max_score_after)], loc ="lower right")
    if saveFolder:
        plt.savefig(saveFolder)
    else:
        plt.show()
    plt.clf()

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
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.0001, 0.99, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.001, 0.90, 'linear', 'mse')
   # ID = run_DQN_CartPole_experiment(ID, 500, 0.001, 0.99, 'linear', 'mse')
   # ID = run_DQN_CartPole_experiment(ID, 500, 0.3, 0.99, 'linear', 'mse')
    ID = run_DQN_CartPole_experiment(ID, 1500, 0.001, 0.9, 'linear', 'mse') # gave good results
   # ID = run_DQN_CartPole_experiment(ID, 500, 0.7, 0.90, 'linear', 'mse')
    #ID = run_DQN_CartPole_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse') #totally fails

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
    ID = run_DQN_FrozenLake_experiment(ID, 1000, 0.001, 0.8, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 500, 0.01, 0.99, 'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_DQN_FrozenLake_experiment(ID, epochs=500, lr=0.001, gamma=0.90, activation='linear', loss='mse'):
    global env
    env = gym.make(FROZENLAKE_ENV_NAME)
    return run_experiment(ID, epochs, lr, gamma, activation, loss, FROZENLAKE_ENV_NAME)

def run_DQN_CartPole_experiment(ID, epochs=500, lr=0.001, gamma=0.90, activation='linear', loss='mse'):
    global env
    env = gym.make(CARTPOLE_ENV_NAME)
    return run_experiment(ID, epochs, lr, gamma, activation, loss, CARTPOLE_ENV_NAME)

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
    ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1, gamma=0.99, result_x_size=50)
    #ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_QT_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
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
    ID = run_QT_cartpole_experiment(ID, epochs=100000, lr=0.3,  gamma=0.99, result_x_size=100)
    #ID = run_QT_cartpole_experiment(ID, epochs=100000, lr=0.7,  gamma=0.99, result_x_size=1000)
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
    #ID = run_dqt_cartpole_experiment(ID, epochs=100000, lr=0.5,  gamma=0.99, result_x_size=100)
    #ID = run_dqt_cartpole_experiment(ID, epochs=100000, lr=0.7,  gamma=0.99, result_x_size=100)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()

def run_DQT_frozenlake_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQT/{0}/models/'.format(FROZENLAKE_ENV_NAME)
    experiment_ID_file = 'experiments/DQT/{0}/exp_ID.txt'.format(FROZENLAKE_ENV_NAME)
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env
    env = gym.make(FROZENLAKE_ENV_NAME)
    model = [QTable.Input(0, 1, 1, 1, static=False)]
    ID = run_dqt_frozenlake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=50, model=model)
    #ID = run_dqt_frozenlake_experiment(ID, epochs=20000, lr=0.1,  gamma=0.99, result_x_size=100)
    #ID = run_dqt_frozenlake_experiment(ID, epochs=20000, lr=0.2,  gamma=0.99, result_x_size=100)
    #ID = run_dqt_frozenlake_experiment(ID, epochs=20000, lr=0.3,  gamma=0.99, result_x_size=100)
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
    min_exploration_rate = 0.001
    exploration_decay_rate = 0.001
    model = [QTable.Input(0, 15, 1, 0, static=True)]
    table = QTable.QTable(env.action_space.n, model=model, dynamic=False)
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
            if done:
                reward = -1 if reward == 0 and s < 99 else reward
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 100)
            state = new_state
            if done:
                reward = 0.0 if reward < 0 else reward
                r = reward
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
        if max_score < r:
            max_score = r
            max_score_after = e
        print("E:", e, "score:", r, "epsilon:", epsilon)
    
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
    max_score_after = np.argmax(rewards)
    max_score = rewards[max_score_after]
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
    starting_model = model.copy()
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
        f.write("Starting intervals: " + str(QTable.QTable(env.action_space.n, model=starting_model, dynamic=True).get_intervals()) + '\n')

        f.write("Intervals: " + str(table.get_intervals()) + '\n')
        print("Final score: " + str(results[len(results)-1]))
        print("Max score: " + str(max_score) + " after " + str(max_score_after) + '\n')
        print("Results saved to", fullPathWithExt)
    table.save(MODELS_FOLDER+str(ID))
    return ID+1
def run_dqt_frozenlake_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100, model=None):
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'DQT/{0}'.format(FROZENLAKE_ENV_NAME)
    folder          = experiments_folder + '/' + agent_folder
    file            = '{0}_'.format(FROZENLAKE_ENV_NAME)+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running DQT FrozenLake experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.0001
    exploration_decay_rate = 0.001
    if model == None:
        model = [QTable.Input(0, 1, 1, 1, static=False)]
    starting_model = model.copy()
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
            if done and reward == 0:
                reward = -1
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 100)
            state = new_state
            #r += reward
            if done:
                r = reward if reward > 0 else 0.0
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
        if max_score < r:
            max_score = r
            max_score_after = e
        print("E:", e, "score:", r, "epsilon:", epsilon)
    
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
        f.write("Starting intervals: " + str(QTable.QTable(env.action_space.n, model=starting_model, dynamic=True).get_intervals()) + '\n')

        f.write("Intervals: " + str(table.get_intervals()) + '\n')
        f.write("Out Of Bounds steps: " + str(table.get_OOB()) + '\n')
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

    agent = DQN_Agent(env, ID, lr, activation, loss, observation_space=4)
    agent.frozen_lake = env_name == FROZENLAKE_ENV_NAME
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

        result_x_size = 10
        rewards, _ = agent.train(gamma=gamma,  epochs=epochs, batchSize=16, file=f)
        max_score_after = np.argmax(rewards)
        max_score = rewards[max_score_after]

        rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
        count = result_x_size

        results = [] # average rewards per result_x_size episodes
        for r in rewards_per_x_episodes:
            results.append(sum(r/result_x_size))
            count += result_x_size

        plot(results, fullPath, ID, xs=[i * result_x_size for i in range(len(results))], x_size=result_x_size, max_score=max_score, max_score_after=max_score_after, lr=lr)
        f.write("Final score: " + str(rewards[len(rewards)-1]) + '\n')
        f.write("Final average score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(rewards[len(rewards)-1]))
        print("Final average score: " + str(results[len(results)-1]))

    return ID+1

if __name__ == "__main__":
    
    start()