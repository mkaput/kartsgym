import copy
import itertools
import logging
import math
import pickle
import random
from collections import defaultdict, deque

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from kartsgym.agents.Agent import Agent
from kartsgym.agents.RewardSystem import RewardSystem


def backet_in_bounds(value, low_bound, high_bound, backets):
    backet = math.ceil((value - low_bound) / (high_bound - low_bound) * backets)
    backet = min(backet, backets)
    backet = max(backet, 0)
    return backet


def get_backet_value(backet, low_bound, high_bound, backets):
    value = (backet * (high_bound - low_bound) / backets) + low_bound
    return value


class DNQLearner(Agent):
    def __init__(self,
                 environment,
                 reward_system=RewardSystem.NORMAL):
        super().__init__(environment, reward_system)

        self.upper_bounds = self.environment.observation_space.high
        self.lower_bounds = self.environment.observation_space.low

        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.alfa = 0.001
        self.batch_size = 32
        self.learning_chance = 0.01
        self.seperate_done = False
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
        self.done_memory = deque(maxlen=10000)

        self.max_attempts = 1
        self.attempt_no = 1

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=len(self.environment.observation_space.high), activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alfa))
        return model

    def calc_eps(self):
        progress = self.attempt_no / float(self.max_attempts)
        if progress < 0.15:
            return 0.5
        if progress < 0.55:
            return 0.1
        if progress < 0.85:
            return 0.05
        return 0.001

    def learn(self, max_attempts, render=False, logs=False):
        self.max_attempts = max_attempts
        return super(DNQLearner, self).learn(max_attempts, render, logs)

    def attempt(self, render=False, logs=False, sleep=0):
        self.attempt_no += 1
        self.epsilon = self.calc_eps()
        return super(DNQLearner, self).attempt(render, logs, sleep)


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
        done_sample = random.sample(self.done_memory, min(self.batch_size, len(self.done_memory)))
        normal_sample = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        trainig_sample = [*done_sample, *normal_sample]

        trainig_observations = np.array([observation for _, observation, _, _, _ in trainig_sample])
        trainig_new_observations = np.array([new_observation for _, _, new_observation, _, _ in trainig_sample])

        training_f = self.model.predict(trainig_observations)
        training_targets = [np.amax(x) for x in self.model.predict(trainig_new_observations)]

        for sample, target_f, target in zip(trainig_sample, training_f, training_targets):
            action, observation, new_observation, reward, done = sample
            if not done:
                target = reward + self.gamma * target
            else:
                target = reward
            target_f[action] = target

        self.model.fit(trainig_observations, training_f, epochs=1, verbose=0, batch_size=len(trainig_observations))

    def update_knowledge(self, action, observation, new_observation, reward, done):
        action = self.action_map[self.discretise_action(action)]
        if done and self.seperate_done:
            self.done_memory.append((action, observation, new_observation, reward, done))
        else:
            self.memory.append((action, observation, new_observation, reward, done))

        if done or random.uniform(0, 1) <= self.learning_chance:
            self.replay()
