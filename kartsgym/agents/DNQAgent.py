import copy
import itertools
import math
import pickle
import random
from collections import defaultdict, deque

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from kartsgym.agents.Agent import Agent


def backet_in_bounds(value, low_bound, high_bound, backets):
    backet = math.ceil((value - low_bound) / (high_bound - low_bound) * backets)
    backet = min(backet, backets)
    backet = max(backet, 0)
    return backet


def get_backet_value(backet, low_bound, high_bound, backets):
    value = (backet * (high_bound - low_bound) / backets) + low_bound
    return value


class DNQLearner(Agent):
    def __init__(self, environment):
        super().__init__(environment)

        self.upper_bounds = self.environment.observation_space.high
        self.lower_bounds = self.environment.observation_space.low

        self.action_backets = 2
        self.action_low_bound = [
            -0.4,
            0
        ]
        self.action_high_bound = [
            0.4,
            0.5
        ]
        self.actions = list(itertools.product(range(self.action_backets + 1), range(self.action_backets + 1)))

        self.action_map = {a: i for i, a in enumerate(self.actions)}

        self.memory = deque(maxlen=10000)

        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.alfa = 0.001
        self.batch_size = 5

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=len(self.environment.observation_space.high), activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alfa))
        return model


    def save_agent(self, filename):
        self.model.save_weights(filename)

    @staticmethod
    def load_agent(filename, env):
        agent = DNQLearner(env)
        agent.model.load_weights(filename)
        return agent

    def discretise(self, observation):
        return tuple(
            (observation[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            for i in range(len(observation))
        )

    def discretise_action(self, action):
        return tuple(
            backet_in_bounds(action[i], self.action_low_bound[i], self.action_high_bound[i], self.action_backets)
            for i in range(len(action))
        )

    def undiscretise_action(self, action):
        return tuple(
            get_backet_value(action[i], self.action_low_bound[i], self.action_high_bound[i], self.action_backets)
            for i in range(len(action))
        )

    def best_action(self, observation):
        q_values = self.model.predict(np.array([observation]))[0]
        action_number = np.argmax(q_values)
        return self.actions[action_number]

    def pick_action(self, observation):
        if not self.eval and random.uniform(0, 1) <= self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.best_action(observation)

        return self.undiscretise_action(action)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for action, observation, new_observation, reward, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(np.array([new_observation]))[0]))
            target_f = self.model.predict(np.array([observation]))
            target_f[0][action] = target
            self.model.fit(np.array([observation]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_knowledge(self, action, observation, new_observation, reward, done):
        action = self.action_map[self.discretise_action(action)]

        self.memory.append((action, observation, new_observation, reward, done))
        if len(self.memory) > self.batch_size:
            self.replay()
