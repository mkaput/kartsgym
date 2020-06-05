import logging
import sys
import uuid
from datetime import datetime

import gym
from kartsgym import environment
from kartsgym.agents.DNQAgent import DNQLearner
from kartsgym.agents.QAgent import QLearner
from kartsgym.agents.RandomAgent import RandomAgent
from kartsgym.agents.RewardSystem import RewardSystem
from kartsgym.agents.RuleAgent import RuleAgent

import numpy as np
import matplotlib.pyplot as plt

import argparse

__version__ = '0.1.0'


def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_to_file(results, filename, ylabel, title, xlabel='Try'):
    fig, ax = plt.subplots()

    n = max(min(len(results) // 10, 50), 1)
    y = np.array(moving_average(results, n=n))
    x = np.array(range(len(y)))
    ax.plot(x, y, 'b-')

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.grid()

    fig.savefig(filename)
    # plt.show()


def metric_learn(results, model_name):
    steps, reward = list(zip(*results))
    reward = [max(x, -1000) for x in reward]

    plot_to_file(steps, f'{model_name}_steps.png', 'steps', f'{model_name} steps')
    plot_to_file(reward, f'{model_name}_rewards.png', 'reward', f'{model_name} reward')


def check_agent(agent, render=True):
    agent.eval = True
    step, reward = agent.attempt(render=render, logs=True, sleep=0)
    logging.info(f"Episode finished after {step} steps with final reward {reward}")
    return step, reward

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


def check_q_agent(alfa, gamma, backets, action_backets, epochs, reward_system):
    env = gym.make('Karts-v0')
    uid = str(uuid.uuid4())[-10:]
    name = f"q-{uid}.pkl"
    agent = QLearner(env, alfa=alfa, gamma=gamma, backets=backets, action_backets=action_backets,
                     reward_system=reward_system)
    start = datetime.now()
    results = agent.learn(epochs, render=False, logs=False)
    end = datetime.now()
    metric_learn(results, name)
    check_agent(agent, False)
    logging.info(
        f"{name}, reward_system={reward_system} epochs={epochs} alfa={alfa}, gamma={gamma}, backets={backets}, action_backets={action_backets}")
    logging.info(f"training time: {(end-start).total_seconds()}")
    agent.save_agent(name)
    env.close()


def check_q_agent_file(filename):
    env = gym.make('Karts-v0')
    agent = QLearner.load_agent(filename, env)
    check_agent(agent)
    env.close()


def check_dnq_agent(epochs, reward_system):
    env = gym.make('Karts-v0')
    uid = str(uuid.uuid4())[-10:]
    name = f"dnq-{uid}.h5"
    agent = DNQLearner(env, reward_system)
    start = datetime.now()
    results = agent.learn(epochs, render=False, logs=False)
    end = datetime.now()
    metric_learn(results, name)
    check_agent(agent, False)
    logging.info(f"{name}, reward_system={reward_system} epochs={epochs}")
    logging.info(f"training time: {(end - start).total_seconds()}")
    agent.save_agent(name)
    env.close()


def check_dnq_agent_file(filename):
    env = gym.make('Karts-v0')
    agent = DNQLearner.load_agent(filename, env)
    check_agent(agent)
    env.close()


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--type", required=True,
                        help="type of agent")

    parser.add_argument("-n", "--name", required=False, default='',
                        help="name of agent to test")

    parser.add_argument("-a", "--alfa", required=False, type=float, default=0.01,
                        help="alfa for QAgent")

    parser.add_argument("-g", "--gamma", required=False, type=float, default=0.8,
                        help="gamma for QAgent")

    parser.add_argument("-b", "--backets", required=False, type=int, default=6,
                        help="observation backets for QAgent")

    parser.add_argument("-ab", "--action_backets", required=False, type=int, default=2,
                        help="observation backets for QAgent")

    parser.add_argument("-e", "--epochs", required=False, type=int, default=100,
                        help="epochs for Agent")

    parser.add_argument("-r", "--reward", required=False, default='NORMAL',
                        help="reward system for Agent")

    args = parser.parse_args()

    if args.type == "Random":
        check_random()
    elif args.type == "Rule":
        check_rule()
    elif args.type == "Q":
        if args.name != '':
            check_q_agent_file(args.name)
        else:
            check_q_agent(
                alfa=args.alfa,
                gamma=args.gamma,
                backets=args.backets,
                action_backets=args.action_backets,
                epochs=args.epochs,
                reward_system=RewardSystem[args.reward]
            )
    elif args.type == "DNQ":
        if args.name != '':
            check_dnq_agent_file(args.name)
        else:
            check_dnq_agent(
                epochs=args.epochs,
                reward_system=RewardSystem[args.reward]
            )
    exit(0)


if __name__ == '__main__':
    main()
