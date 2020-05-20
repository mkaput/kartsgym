import gym
from kartsgym import environment
from kartsgym.agents.QAgent import QLearner
from kartsgym.agents.RandomAgent import RandomAgent

__version__ = '0.1.0'


def check_agent(agent):
    step, reward = agent.attempt(render=True, logs=True)
    print(f"Episode finished after {step} steps with final reward {reward}")

def check_random():
    env = gym.make('Karts-v0')
    agent = RandomAgent(env)
    check_agent(agent)
    env.close()

def check_q_agent():
    env = gym.make('Karts-v0')
    agent = QLearner(env, alfa=0.1, gamma=0.8, backets=10, action_backets=4)
    agent.learn(1000)
    check_agent(agent)
    agent.save_agent("sample3.pkl")
    env.close()

def check_q_agent_file():
    env = gym.make('Karts-v0')
    agent = QLearner.load_agent("sample3.pkl", env)
    check_agent(agent)
    env.close()

def main():

    # check_random()
    check_q_agent()
    check_q_agent_file()
    exit(0)


if __name__ == '__main__':
    main()
