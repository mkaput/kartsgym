import logging
import sys

import gym
from kartsgym import environment
from kartsgym.agents.QAgent import QLearner
from kartsgym.agents.RandomAgent import RandomAgent

__version__ = '0.1.0'


def check_agent(agent):
    step, reward = agent.attempt(render=True, logs=True)
    logging.info(f"Episode finished after {step} steps with final reward {reward}")

def check_random():
    env = gym.make('Karts-v0')
    agent = RandomAgent(env)
    check_agent(agent)
    env.close()

def check_q_agent():
    env = gym.make('Karts-v0')
    agent = QLearner(env, alfa=0.1, gamma=0.8, backets=10, action_backets=10)
    agent.learn(2000)
    check_agent(agent)
    agent.save_agent("sample.pkl")
    env.close()

def check_q_agent_file():
    env = gym.make('Karts-v0')
    agent = QLearner.load_agent("sample.pkl", env)
    check_agent(agent)
    env.close()

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s  %(message)s', stream=sys.stdout, level=logging.DEBUG)
    # check_random()
    check_q_agent()
    # check_q_agent_file()
    exit(0)


if __name__ == '__main__':
    main()
