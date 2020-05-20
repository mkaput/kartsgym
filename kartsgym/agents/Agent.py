import abc
import logging


class Agent:
    def __init__(self, environment):
        self.environment = environment

    def attempt(self, render=False, logs=False):
        observation = self.discretise(self.environment.reset())
        done = False
        step = 0
        reward = None
        while not done:
            if render:
                self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            if logs:
                logging.debug(observation, '=>', action, '=>', reward, done)
            self.update_knowledge(action, observation, new_observation, reward)
            step += 1
            observation = new_observation
        return step, reward

    def learn(self, max_attempts, render=False):
        results = []
        for i in range(max_attempts):
            logging.info(f"start {i}")
            results.append(self.attempt(render=render))
            logging.info(f"end {i} with results: {results[i]}")
        return results

    def discretise(self, observation):
        """Dicretise observation if needed"""
        return observation

    @abc.abstractmethod
    def pick_action(self, observation):
        """Pick agent action base on environment"""

    def update_knowledge(self, action, observation, new_observation, reward):
        """Update agent knowledge based on reward"""
        pass

    def reset_knowledge(self):
        """Reset agent knowledge"""
        pass
