from dataclasses import dataclass, field
from importlib import resources
from typing import List, Tuple
from xml.etree import ElementTree
from xml.etree.ElementTree import QName

from itertools import chain

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class Map:
    kart: Point = field(default=(0.0, 0.0))
    barriers: List[Polygon] = field(default_factory=list)
    checkpoints: List[Polygon] = field(default_factory=list)

    def dim(self) -> Tuple[float, float, float, float]:
        minx = min(x for x, _ in chain(*self.barriers, *self.checkpoints))
        maxx = max(x for x, _ in chain(*self.barriers, *self.checkpoints))
        miny = min(y for _, y in chain(*self.barriers, *self.checkpoints))
        maxy = max(y for y, y in chain(*self.barriers, *self.checkpoints))
        return minx, maxx, miny, maxy

    @classmethod
    def load(cls, name: str) -> 'Map':
        def q(tag):
            return QName('http://www.w3.org/2000/svg', tag)

        with resources.open_binary(cls.__module__, f'{name}.svg') as f:
            tree = ElementTree.parse(f)

        m = cls()
        barriers = []
        checkpoints = []

        for n, elem in enumerate(tree.getroot()):
            if elem.attrib.get('class') in ['checkpoint', 'barrier']:
                if elem.attrib['class'] == 'checkpoint':
                    target = checkpoints
                else:
                    target = barriers

                if elem.tag == q('polygon'):
                    poly = []
                    polyid = elem.attrib.get('id', str(n))

                    for point_str in elem.attrib['points'].split():
                        [x_str, y_str] = point_str.split(',')
                        point = float(x_str), -float(y_str)
                        poly.append(point)

                    # Remove loops, they will be handled by Box2D
                    if len(poly) > 1 and poly[0] == poly[-1]:
                        poly.pop()

                    target.append((poly, polyid))
                else:
                    raise ValueError(f'{elem.attrib["class"]} must be a polygon but is {elem.tag}')

        barriers.sort(key=lambda p: p[1])
        checkpoints.sort(key=lambda p: p[1])

        m.barriers = [p for p, _ in barriers]
        m.checkpoints = [p for p, _ in checkpoints]

        starting_checkpoint = m.checkpoints[0]
        kart_x = sum(x for x, _ in starting_checkpoint) / len(starting_checkpoint)
        kart_y = sum(y for _, y in starting_checkpoint) / len(starting_checkpoint)
        m.kart = kart_x, kart_y

        return m
