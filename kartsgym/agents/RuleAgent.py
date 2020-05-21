from kartsgym.agents.Agent import Agent


class RuleAgent(Agent):
    def __init__(self, environment):
        super().__init__(environment)

    def pick_action(self, observation):
        if observation[5] > 100:
            return 0.5, 0.3
        if observation[4] > 90 and all(obs > 40 for obs in observation[2:]):
            return 0, 0.3
        elif observation[6] > 70 or observation[5] > 70:
            return 0.5, 0.3
        else:
            return -0.5, 0.3

    def learn(self, max_attempts, render=False, logs=False):
        return []
