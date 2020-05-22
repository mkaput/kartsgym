import abc
import logging
import time


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.eval = False

    def attempt(self, render=False, logs=False, sleep=0):
        observation = self.discretise(self.environment.reset())
        done = False
        step = 0
        reward = None
        if render:
            self.environment.render()
            time.sleep(sleep)
        while not done:
            if render:
                self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            if logs:
                logging.debug(f"step: {step} {observation} => {action} => {reward} {done}")
            if not self.eval:
                self.update_knowledge(action, observation, new_observation, reward, done)
            step += 1
            observation = new_observation
        return step, reward

    def learn(self, max_attempts, render=False, logs=False):
        results = []
        for i in range(max_attempts):
            logging.info(f"start try {i}")
            results.append(self.attempt(render=render, logs=logs))
            logging.info(f"end {i} with results: steps: {results[i][0]}, reward: {results[i][1]}")
        return results

    def discretise(self, observation):
        """Dicretise observation if needed"""
        return observation

    @abc.abstractmethod
    def pick_action(self, observation):
        """Pick agent action base on environment"""

    def update_knowledge(self, action, observation, new_observation, reward, done):
        """Update agent knowledge based on reward"""
        pass

    def reset_knowledge(self):
        """Reset agent knowledge"""
        pass
