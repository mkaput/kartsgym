from . import environment

__version__ = '0.1.0'


def main():
    import gym
    env = gym.make('Karts-v0')

    for i_episode in range(3):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation, '=>', action, '=>', reward, done)
            if done:
                print(f"Episode finished after {t + 1} timesteps")
                break
        else:
            print("Episode failed")

    env.close()
    exit(0)


if __name__ == '__main__':
    main()
