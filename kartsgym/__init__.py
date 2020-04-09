from . import environment

__version__ = '0.1.0'


def main():
    import gym
    env = gym.make('Karts-v0')

    for i_episode in range(3):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()


if __name__ == '__main__':
    main()
