from kartsgym.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, environment):
        super().__init__(environment)

    def pick_action(self, observation):
        return self.environment.action_space.sample()

    def learn(self, max_attempts, render=False):
        return []