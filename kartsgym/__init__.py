import logging
import sys
import uuid

import gym
from kartsgym import environment
from kartsgym.agents.QAgent import QLearner
from kartsgym.agents.RandomAgent import RandomAgent
from kartsgym.agents.RuleAgent import RuleAgent

__version__ = '0.1.0'


def check_agent(agent):
    agent.eval = True
    step, reward = agent.attempt(render=True, logs=True, sleep=5)
    logging.info(f"Episode finished after {step} steps with final reward {reward}")

def check_random():
    env = gym.make('Karts-v0')
    agent = RandomAgent(env)
    check_agent(agent)
    env.close()

def check_rule():
    env = gym.make('Karts-v0')
    agent = RuleAgent(env)
    check_agent(agent)
    env.close()

def check_q_agent():
    env = gym.make('Karts-v0')
    agent = QLearner(env, alfa=0.01, gamma=0.8, backets=6, action_backets=2)
    agent.learn(10000, render=False, logs=False)
    check_agent(agent)
    uid = str(uuid.uuid4())[-10:]
    name = f"q-{uid}.pkl"
    logging.info(name)
    agent.save_agent(name)
    env.close()

def check_q_agent_file():
    env = gym.make('Karts-v0')
    agent = QLearner.load_agent("q-d40cf2957b-copy-bad.pkl", env)
    check_agent(agent)
    env.close()

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout, level=logging.DEBUG)
    # check_rule()
    # check_random()
    # check_q_agent()
    check_q_agent_file()
    exit(0)

if __name__ == '__main__':
    main()
