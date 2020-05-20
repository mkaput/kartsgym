import copy
import itertools
import math
import random
from collections import defaultdict

from kartsgym.agents.Agent import Agent


def backet_in_bounds(value, low_bound, high_bound, backets):
    backet = math.ceil((value - low_bound) / (high_bound - low_bound) * backets)
    backet = min(backet, backets)
    backet = max(backet, 0)
    return backet


def get_backet_value(backet, low_bound, high_bound, backets):
    value = (backet * (high_bound - low_bound) / backets) + low_bound
    return value


class QLearner(Agent):
    def __init__(self, environment, alfa, gamma, backets, action_backets):
        super().__init__(environment)
        self.backets = backets
        self.upper_bounds = self.environment.observation_space.high
        self.lower_bounds = self.environment.observation_space.low

        self.action_backets = action_backets
        self.action_low_bound = self.environment.action_space.low
        self.action_high_bound = self.environment.action_space.high
        self.actions = list(itertools.product(range(action_backets + 1), range(action_backets + 1)))

        self.alfa = alfa
        self.gamma = gamma
        self.Q = None

        self.attempt_no = 0
        self.max_attempts = 1
        self.reset()

    def reset(self, max_attempts=1):
        default_value = {a: 0 for a in self.actions}
        self.Q = defaultdict(lambda: copy.deepcopy(default_value))
        self.attempt_no = 0
        self.max_attempts = max_attempts

    def learn(self, max_attempts, render=False):
        self.reset(max_attempts)
        return super(QLearner, self).learn(max_attempts, render)

    def attempt(self, render=False, logs=False):
        self.attempt_no += 1
        return super(QLearner, self).attempt(render, logs)

    def discretise(self, observation):
        return tuple(
            backet_in_bounds(observation[i], self.lower_bounds[i], self.upper_bounds[i], self.backets)
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
        return max(self.Q[tuple(observation)].items(), key=lambda x: x[1])[0]

    def calc_eps(self):
        progress = self.attempt_no / float(self.max_attempts)
        if progress < 0.15:
            return 0.5
        if progress < 0.55:
            return 0.1
        if progress < 0.85:
            return 0.01
        return 0.001

    def pick_action(self, observation):
        if random.uniform(0, 1) <= self.calc_eps():
            action = random.choice(self.actions)
        else:
            action = self.best_action(observation)

        return self.undiscretise_action(action)

    def update_knowledge(self, action, observation, new_observation, reward):
        # reward = reward if reward < -200 else 0
        action = self.discretise_action(action)
        next_state_reward = self.Q[new_observation][self.best_action(new_observation)]

        self.Q[observation][action] = (1 - self.alfa) * self.Q[observation][action] + self.alfa * (
                reward + self.gamma * next_state_reward)
