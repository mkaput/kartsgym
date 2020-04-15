# Based on: https://www.iforce2d.net/b2dtut/top-down-car

import weakref
from dataclasses import dataclass, InitVar, field
from typing import List, Tuple

import numpy as np
from Box2D import b2Body, b2World, b2CircleShape, b2PolygonShape, b2Vec2, b2Dot, \
    b2RevoluteJoint, b2EdgeShape, b2ContactListener
from gym import Env, register, spaces
from itertools import tee, cycle

from kartsgym.map import Map

FPS = 50

VIEWPORT_W = 1024
VIEWPORT_H = 1024


def hexcol(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


BARRIER = 'barrier'
CHASSIS = 'chassis'
CHECKPOINT = 'checkpoint'
WHEEL = 'wheel'

COLOR_BACKGROUND = hexcol('FFFFFF')
COLOR_KART_CHASSIS = hexcol('d50000'), hexcol('311B92')
COLOR_KART_WHEEL = hexcol('212121'), hexcol('000000')
COLOR_BARRIER = hexcol('b71c1c'),
COLOR_TRACK_VISITED = hexcol('F1F8E9'), hexcol('DCEDC8')
COLOR_TRACK_NOT_VISITED = hexcol('FAFAFA'), hexcol('F5F5F5')


def pairwise(it):
    a, b = tee(it)
    next(b, None)
    return zip(a, b)


@dataclass
class Wheel:
    world: InitVar[b2World]

    max_drive_force: float
    max_brake_force: float
    max_lateral_impulse: float

    traction: float = 1.0
    position: InitVar[b2Vec2] = b2Vec2(0.0, 0.0)

    body: b2Body = field(init=False)

    def __post_init__(self, world: b2World, position):
        self.body: b2Body = world.CreateDynamicBody(position=position, userData=(WHEEL, None))
        self.body.CreatePolygonFixture(box=(0.5, 1.25), density=1.0)
        self.body.color = COLOR_KART_WHEEL

    @property
    def lateral_velocity(self) -> b2Vec2:
        current_right_normal = self.body.GetWorldVector(b2Vec2(1, 0))
        return b2Dot(current_right_normal, self.body.linearVelocity) * current_right_normal

    @property
    def forward_velocity(self) -> b2Vec2:
        current_forward_normal = self.body.GetWorldVector(b2Vec2(0, 1))
        return b2Dot(current_forward_normal, self.body.linearVelocity) * current_forward_normal

    def update_friction(self):
        impulse = self.body.mass * -self.lateral_velocity
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length
        self.body.ApplyLinearImpulse(self.traction * impulse, self.body.worldCenter, True)

        self.body.ApplyAngularImpulse(
            self.traction * 0.1 * self.body.inertia * self.body.angularVelocity, True)

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()
        drag_force_magnitude = -2.0 * current_forward_speed
        self.body.ApplyForce(self.traction * drag_force_magnitude * current_forward_normal,
                             self.body.worldCenter, True)

    def update_drive(self, gas):
        gas = np.clip(gas, self.max_brake_force, self.max_drive_force)
        self.body.ApplyForce(self.traction * gas * self.body.GetWorldVector(b2Vec2(0, 1)),
                             self.body.worldCenter, True)

    def draw_chain(self):
        yield self.body


@dataclass
class Kart:
    world: InitVar[b2World]
    position: InitVar[Tuple[float, float]]

    body: b2Body = field(init=False)
    joints: List[b2RevoluteJoint] = field(init=False, default_factory=lambda: [None] * 4)
    wheels: List[Wheel] = field(init=False, default_factory=lambda: [None] * 4)

    def __post_init__(self, world, position):
        chassis_position = b2Vec2(position)
        self.body = world.CreateDynamicBody(position=chassis_position, angularDamping=3,
                                            userData=(CHASSIS, None))
        self.body.CreatePolygonFixture(vertices=[
            (1.5, 0),
            (3, 2.5),
            (2.8, 5.5),
            (1, 10),
            (-1, 10),
            (-2.8, 5.5),
            (-3, 2.5),
            (-1.5, 0),
        ], density=1.0)
        self.body.color = COLOR_KART_CHASSIS

        for i, (x, y, is_front) in enumerate([
            (-3, 8.5, True),
            (3, 8.5, True),
            (-3, 0.75, False),
            (3, 0.75, False),
        ]):
            if is_front:
                max_drive_force = 500
                max_brake_force = -150
                max_lateral_impulse = 7.5
            else:
                max_drive_force = 300
                max_brake_force = -150
                max_lateral_impulse = 8.5

            wheel_position = b2Vec2(x, y) + chassis_position
            self.wheels[i] = Wheel(world, max_drive_force, max_brake_force, max_lateral_impulse,
                                   position=wheel_position)
            self.joints[i] = world.CreateRevoluteJoint(
                enableLimit=True,
                lowerAngle=0.0,
                upperAngle=0.0,
                bodyA=self.body,
                bodyB=self.wheels[i].body,
                localAnchorA=b2Vec2(x, y),
                localAnchorB=b2Vec2(0, 0),
                maxMotorTorque=1000,
            )

    def update(self, wheel_angle: float, gas: float):
        wheel_angle = np.clip(wheel_angle, -35.0, 35.0)
        gas = np.clip(gas, -1.0, 1.0)
        if gas >= 0.0:
            gas *= self.wheels[0].max_drive_force
        else:
            gas *= -self.wheels[0].max_brake_force

        for wheel in self.wheels:
            wheel.update_friction()
        for wheel in self.wheels:
            wheel.update_drive(gas)

        for joint in self.joints[:2]:
            joint.SetLimits(wheel_angle, wheel_angle)

    def draw_chain(self):
        yield self.body
        for wheel in self.wheels:
            yield from wheel.draw_chain()


class ContactListener(b2ContactListener):
    def __init__(self, world: 'World'):
        super().__init__()
        self.on_contact = weakref.WeakMethod(world.on_contact)

    def BeginContact(self, contact):
        self.on_contact()(contact, True)

    def EndContact(self, contact):
        self.on_contact()(contact, False)


class World:
    def __init__(self, world_map: Map):
        self.dim = world_map.dim()
        self.contact_listener = ContactListener(self)
        self.world = b2World(gravity=(0, 0), contactListener=self.contact_listener)

        self.barriers = []
        for loop in world_map.barriers:
            barrier = self.world.CreateStaticBody(userData=(BARRIER, None))
            barrier.CreateEdgeChain(loop)
            barrier.color = COLOR_BARRIER
            self.barriers.append(barrier)

        self.checkpoints = []
        for i, loop in enumerate(world_map.checkpoints):
            checkpoint = self.world.CreateStaticBody(userData=(CHECKPOINT, i))
            checkpoint.CreatePolygonFixture(vertices=loop, isSensor=True)
            checkpoint.visited = False
            checkpoint.color = COLOR_TRACK_NOT_VISITED
            self.checkpoints.append(checkpoint)

        self.kart = Kart(self.world, world_map.kart)

        self.hit_barriers = False
        self.elapsed_steps = 0

    @property
    def checkpoint_discontinuity(self):
        v = True
        for c in self.checkpoints:
            if c.visited > v:
                return True
            v = c.visited
        return False

    def step(self, wheel_angle: np.float32, gas: np.float32):
        self.kart.update(wheel_angle, gas)
        self.world.Step(1 / FPS, 6 * 30, 2 * 30)
        self.elapsed_steps += 1

    def on_contact(self, contact, began):
        fixture_a, fixture_b = contact.fixtureA, contact.fixtureB
        if fixture_a.body.userData > fixture_b.body.userData:
            fixture_a, fixture_b = fixture_b, fixture_a

        body_a, body_b = fixture_a.body, fixture_b.body
        ud_a, ud_b = body_a.userData, body_b.userData
        if not ud_a or not ud_b:
            return

        ty_a, opt_a = ud_a
        ty_b, opt_b = ud_b

        if ty_a is CHECKPOINT and ty_b is WHEEL and began:
            checkpoint = self.checkpoints[opt_a]
            checkpoint.visited = True
            checkpoint.color = COLOR_TRACK_VISITED
        elif ty_a is BARRIER and ty_b is WHEEL and began:
            self.hit_barriers = True

    def draw_chain(self):
        yield from self.checkpoints
        yield from self.barriers
        yield from self.kart.draw_chain()


class KartsEnv(Env):
    """
    Karts time run simulation environment.

    The whole simulation happens in a (0,0)-centered, metre-based, coordinate system
    using np.float32 for all metric variables. The exact size of the world is determined
    by the chosen map.

    Action space

    1. Wheel, full left being -35deg, straight 0deg, full right 35deg
    2. Gas&Brake, no gas being 0.0, full gas being 1.0, full brake being -1.0

    Observation space

    1. Current speed in metres/second, from 0.0 (meaning stop) to infinity.
    2-6. The distance to the closest obstacle in metres, observed at angles
         relative to the front of the kart: -90deg, -45deg, 0deg, 45deg, 90deg.
         Where 0.0 means direct collision.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, world_map: str = 'circle'):
        super().__init__()

        self.viewer = None
        self.map = Map.load(world_map)
        self.world = World(self.map)

        self.action_space = spaces.Box(
            low=np.array([np.deg2rad(-35.0), -1.0]),
            high=np.array([np.deg2rad(35.0), 1.0]),
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,))

    def reset(self):
        self.world = World(self.map)
        return self.observe()

    def step(self, action: List[np.float32]):
        self.world.step(action[0], action[1])
        return self.observe(), self.total_reward(), self.is_done(), None

    def render(self, mode='human'):
        import gym.envs.classic_control.rendering as r
        if self.viewer is None:
            self.viewer = r.Viewer(VIEWPORT_W, VIEWPORT_H)

        minx, maxx, miny, maxy = self.world.dim

        width = maxx - minx
        height = maxy - miny

        # Maintain 1:1 aspect ratio
        if width > height:
            diff = (width - height) / 2.0
            miny -= diff
            maxy += diff
        elif height > width:
            diff = (height - width) / 2.0
            minx -= diff
            maxx += diff

        # Add some margins
        margin = 10
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin

        self.viewer.set_bounds(minx, maxx, miny, maxy)
        self.viewer.draw_polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)],
                                 color=COLOR_BACKGROUND)

        for obj in self.world.draw_chain():
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = r.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(radius=f.shape.radius, color=obj.color[0]).add_attr(t)
                    self.viewer.draw_circle(radius=f.shape.radius, color=obj.color[1],
                                            filled=False).add_attr(t)
                    self.viewer.draw_polyline([(0, 0), (0, f.shape.radius)], color=obj.color[1],
                                              linewidth=3).add_attr(t)
                elif type(f.shape) is b2PolygonShape:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color[0])
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color[1])
                elif type(f.shape) is b2EdgeShape:
                    path = [trans * v for v in f.shape.vertices]
                    for (a, b), color in zip(pairwise(path), cycle(obj.color)):
                        self.viewer.draw_polyline([a, b], color=color, linewidth=3)
                else:
                    raise TypeError(f'Unknown shape to draw: {type(f.shape)}')

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def observe(self):
        velocity = self.world.kart.body.linearVelocity.length
        return [velocity, 0.0, 0.0, 0.0, 0.0, 0.0]

    def total_reward(self) -> float:
        if self.world.hit_barriers or self.world.checkpoint_discontinuity:
            return -1000000.0

        reward = -float(self.world.elapsed_steps)
        for checkpoint in self.world.checkpoints:
            if checkpoint.visited:
                reward += 250.0
        return reward

    def is_done(self) -> bool:
        return self.world.hit_barriers or self.world.checkpoint_discontinuity \
               or all(c.visited for c in self.world.checkpoints)


register(
    id='Karts-v0',
    entry_point=f'{KartsEnv.__module__}:{KartsEnv.__name__}'
)
