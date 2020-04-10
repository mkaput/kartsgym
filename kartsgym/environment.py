from typing import List

import numpy as np
from Box2D import b2Body, b2World, b2FixtureDef, b2CircleShape
from gym import Env, register, spaces

from kartsgym.map import Map

FPS = 50

VIEWPORT_W = 640
VIEWPORT_H = 640

KART_MASS = 300
KART_RADIUS = 1.5

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


class World:
    def __init__(self, map: Map):
        self.width = 50.0
        self.height = 50.0
        self.world = b2World(gravity=(0, 0))

        self.kart: b2Body = self.world.CreateDynamicBody(
            position=(3.0, -1.0),
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=KART_RADIUS),
                density=1,
            ),
        )
        self.kart.color1, self.kart.color2 = BLUE, BASE03

    def step(self, wheel_angle: np.float32, gas: np.float32):
        self.world.Step(1 / FPS, 6 * 30, 2 * 30)

    def draw_list(self):
        return [
            self.kart,
        ]


class KartsEnv(Env):
    """
    Karts time run simulation environment.

    The whole simulation happens in a (0,0)-centered, metre-based, coordinate system
    using np.float32 for all metric variables. The exact size of the world is determined
    by the chosen map.

    Action space

    1. Wheel, full left being -100deg, straight 0deg, full right 100deg
    2. Gas, no gas being 0.0, full gas being 1.0

    Observation space

    1. Current speed, from 0.0 (meaning stop) to infinity.
    2-6. The distance to the closest obstacle expressed as the distance in world's coordinate
        system, seen at angles relative to the front of the kart:
        -90deg, -45deg, 0deg, 45deg, 90deg. Where 0.0 means direct contact.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KartsEnv, self).__init__()

        self.viewer = None
        self.map = NotImplemented
        self.world = World(self.map)

        self.action_space = spaces.Box(
            low=np.array([np.deg2rad(-100.0), 0.0]),
            high=np.array([np.deg2rad(100.0), 1.0]),
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,))

    def reset(self):
        self.world = World(self.map)
        return self.observe()

    def step(self, action: List[np.float32]):
        self.world.step(action[0], action[1])
        return self.observe(), self.reward(), self.is_done(), None

    def render(self, mode='human'):
        import gym.envs.classic_control.rendering as r
        if self.viewer is None:
            self.viewer = r.Viewer(VIEWPORT_W, VIEWPORT_H)

        bounds = (
            -self.world.width / 2,
            self.world.width / 2,
            -self.world.height / 2,
            self.world.height / 2,
        )
        self.viewer.set_bounds(*bounds)
        self.viewer.draw_polygon([
            (bounds[0], bounds[3]),
            (bounds[1], bounds[3]),
            (bounds[1], bounds[2]),
            (bounds[0], bounds[2]),
        ], color=BASE3)

        for obj in self.world.draw_list():
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = r.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(radius=f.shape.radius, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(radius=f.shape.radius, color=obj.color2,
                                            filled=False).add_attr(t)
                    self.viewer.draw_polyline([(0, 0), (0, f.shape.radius)], color=obj.color2,
                                              linewidth=3).add_attr(t)
                else:
                    raise TypeError(f'Unknown shape to draw: {type(f.shape)}')

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def observe(self):
        # TODO
        return NotImplemented

    def reward(self) -> np.float32:
        # TODO
        return NotImplemented

    def is_done(self) -> bool:
        # TODO
        return False


register(
    id='Karts-v0',
    entry_point=f'{KartsEnv.__module__}:{KartsEnv.__name__}'
)
