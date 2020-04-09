import numpy as np
import pymunk
from gym import Env, register, spaces

FPS = 50

VIEWPORT_W = 640
VIEWPORT_H = 640

# Solarized Colors
BASE03 = (0 / 255, 43 / 255, 54 / 255)
BASE02 = (7 / 255, 54 / 255, 66 / 255)
BASE01 = (88 / 255, 110 / 255, 117 / 255)
BASE00 = (101 / 255, 123 / 255, 131 / 255)
BASE0 = (131 / 255, 148 / 255, 150 / 255)
BASE1 = (147 / 255, 161 / 255, 161 / 255)
BASE2 = (238 / 255, 232 / 255, 213 / 255)
BASE3 = (253 / 255, 246 / 255, 227 / 255)
YELLOW = (181 / 255, 137 / 255, 0 / 255)
ORANGE = (203 / 255, 75 / 255, 22 / 255)
RED = (220 / 255, 50 / 255, 47 / 255)
MAGENTA = (211 / 255, 54 / 255, 130 / 255)
VIOLET = (108 / 255, 113 / 255, 196 / 255)
BLUE = (38 / 255, 139 / 255, 210 / 255)
CYAN = (42 / 255, 161 / 255, 152 / 255)
GREEN = (133 / 255, 153 / 255, 0 / 255)


class KartsEnv(Env):
    """
    Karts time run simulation environment.

    The whole simulation happens in a (0,0) to (1,1) coordinate system using np.float32
    for all variables.

    Action space

    1. Wheel, full left being -100deg, straight 0deg, full right 100deg
    2. Gas, no gas being 0.0, full gas being 1.0

    Observation space

    1. Current speed, from 0.0 (meaning stop) to infinity.
    2-6. The distance to the closest obstacle expressed as the distance in world's coordinate
        system, seen at angles relative to the front of the kart:
        -90deg, -45deg, 0deg, 45deg, 90deg. Where 0.0 means direct contact.
    """

    space: pymunk.Space

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KartsEnv, self).__init__()

        self.viewer = None

        self.reset()

        self.action_space = spaces.Box(
            low=np.array([np.deg2rad(-100.0), 0.0], dtype=np.float32),
            high=np.array([np.deg2rad(100.0), 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(np.float32(0.0), np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.space = pymunk.Space()

        # TODO

        return self.observe()

    def step(self, action):
        # TODO

        self.space.step(1/FPS)

        return self.observe(), 0.0, False, None

    def render(self, mode='human'):
        from gym.envs.classic_control.rendering import Viewer
        if self.viewer is None:
            self.viewer = Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(0.0, 1.0, 0.0, 1.0)

        self.viewer.draw_polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], color=BASE3)

        # TODO

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def observe(self):
        # TODO
        return NotImplemented


register(
    id='Karts-v0',
    entry_point=f'{KartsEnv.__module__}:{KartsEnv.__name__}'
)
