from kartsgym import environment
from kartsgym.agents.QAgent import QLearner
from kartsgym.agents.RandomAgent import RandomAgent

__version__ = '0.1.0'


def main():
    import gym
    env = gym.make('Karts-v0')

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # print(env.action_space.high)

    agent = QLearner(env, alfa=0.1, gamma=0.8, backets=10, action_backets=10)

    agent.learn(1000)

    for i_episode in range(1):
        step, reward = agent.attempt(render=True, logs=True)
        print(f"Episode finished after {step} steps with final reward {reward}")

    env.close()
    exit(0)


if __name__ == '__main__':
    main()
