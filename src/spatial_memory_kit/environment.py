"""Small continuous 2-D world with a 16x16 egocentric grayscale camera.

Coordinates, layout and object identities belong to the simulator/evaluator.
The agent receives ONLY image, bilateral contact, local motion and energy.
"""
from dataclasses import dataclass
import copy
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    size: float = 10.0
    radius: float = 0.16
    step_size: float = 0.18
    turn_angle: float = np.pi / 12
    vision_size: int = 16
    fov: float = 2 * np.pi / 3


class World:
    def __init__(self, layout_seed=0, config=None):
        self.config = config or WorldConfig()
        self.layout_seed = int(layout_seed)
        rng = np.random.default_rng(layout_seed)
        s = self.config.size
        # Boundary and internal rectangular solids. Texture is object material,
        # sampled independently of absolute coordinates and compass direction.
        rectangles = [[-1, -1, 0, s+1], [s, -1, s+1, s+1],
                      [0, -1, s, 0], [0, s, s, s+1]]
        for _ in range(4):
            x, y = rng.uniform(1.5, s-2.5, 2)
            w, h = rng.uniform(.35, 1.6, 2)
            rectangles.append([x, y, x+w, y+h])
        self.rectangles = np.asarray(rectangles, dtype=np.float32)
        self.material = rng.uniform(.35, .85, len(rectangles))
        self.phase = rng.uniform(0, 2*np.pi, len(rectangles))
        self.light = rng.uniform(1, s-1, 2)
        self.rng = rng
        self.initial_food = np.asarray([self._free_position(rng) for _ in range(6)])
        self.reset(0)

    def collides(self, point):
        r = self.config.radius
        closest = np.clip(point, self.rectangles[:, :2], self.rectangles[:, 2:])
        return bool(np.any(np.sum((closest-point)**2, axis=1) <= r*r))

    def _free_position(self, rng):
        for _ in range(10000):
            point = rng.uniform(.3, self.config.size-.3, 2)
            if not self.collides(point):
                return point
        raise RuntimeError('No free start in world')

    def reset(self, episode_seed):
        self.rng = np.random.default_rng(episode_seed)
        self.position = self._free_position(self.rng)
        self.angle = float(self.rng.uniform(-np.pi, np.pi))
        self.energy = float(self.rng.uniform(.65, 1))
        self.food = self.initial_food.copy()
        self.contact = np.zeros(2, np.float32)
        self.speed = self.rotation = 0.0
        self.steps = self.eaten = 0
        return self.observe()

    def clone(self):
        return copy.deepcopy(self)

    def step(self, action):
        if int(action) not in range(4):
            raise ValueError('action must be 0,1,2,3')
        self.contact[:] = 0
        self.speed = self.rotation = 0.0
        if action == 0:
            direction = np.array([np.cos(self.angle), np.sin(self.angle)])
            candidate = self.position + direction*self.config.step_size
            if self.collides(candidate):
                # Two local whiskers, no external coordinates in observation.
                for side, offset in enumerate([-.45, .45]):
                    probe = self.position + self.config.step_size*np.array(
                        [np.cos(self.angle+offset), np.sin(self.angle+offset)])
                    self.contact[side] = self.collides(probe)
                if not self.contact.any():
                    self.contact[:] = 1
            else:
                self.position = candidate
                self.speed = 1.0
        elif action in (1, 2):
            self.rotation = -1.0 if action == 1 else 1.0
            self.angle = float((self.angle + self.rotation*self.config.turn_angle + np.pi) % (2*np.pi)-np.pi)
        self.energy = max(0.0, self.energy-.001-.0005*(action == 0))
        if len(self.food):
            eaten = np.linalg.norm(self.food-self.position, axis=1) < .38
            self.eaten += int(eaten.sum())
            self.energy = min(1.0, self.energy+.2*eaten.sum())
            self.food = self.food[~eaten]
        self.steps += 1
        # No terminal energy condition in H1/H2 collection (fixed horizon).
        return self.observe()

    def camera(self):
        n = self.config.vision_size
        offsets = np.linspace(-self.config.fov/2, self.config.fov/2, n)
        rays = np.column_stack([np.cos(self.angle+offsets), np.sin(self.angle+offsets)])
        safe = np.where(np.abs(rays) < 1e-8, 1e-8, rays)
        a = (self.rectangles[None, :, :2]-self.position)/safe[:, None, :]
        b = (self.rectangles[None, :, 2:]-self.position)/safe[:, None, :]
        near = np.minimum(a, b).max(axis=2)
        far = np.maximum(a, b).min(axis=2)
        distances = np.where((far >= near) & (far > 0), np.maximum(near, 0), np.inf)
        surface = distances.argmin(axis=1)
        depth = np.maximum(distances[np.arange(n), surface], .05)
        hit = self.position + rays*depth[:, None]
        light_level = .35 + .65/(1 + .08*np.sum((hit-self.light)**2, axis=1))
        texture = .8 + .2*np.sin(3.1*hit.sum(axis=1)+self.phase[surface])
        shade = self.material[surface]*light_level*texture/(1+.045*depth)
        # Perspective projection of vertical walls into a tiny camera.
        half_height = np.minimum(n, 6/np.maximum(depth*np.cos(offsets), .15))
        rows = np.arange(n)[:, None] + .5
        wall = np.abs(rows-n/2) <= half_height[None, :]
        image = np.broadcast_to(np.where(rows < n/2, .08, .17), (n, n)).copy()
        image[wall] = np.broadcast_to(shade, (n, n))[wall]
        # Food appears as a small bright patch only when visible and unoccluded.
        for food in self.food:
            delta = food-self.position
            distance = np.linalg.norm(delta)
            bearing = (np.arctan2(delta[1], delta[0])-self.angle+np.pi)%(2*np.pi)-np.pi
            if abs(bearing) <= self.config.fov/2:
                col = int(np.clip(round((bearing/self.config.fov+.5)*(n-1)), 0, n-1))
                if distance < depth[col]:
                    row = int(np.clip(n/2 + 2.3/max(distance, .3), 0, n-1))
                    image[max(0,row-1):min(n,row+1), max(0,col-1):min(n,col+1)] = .95
        return image.astype(np.float32)[None]

    def observe(self):
        return {'image': self.camera(), 'body': np.array(
            [*self.contact, self.speed, self.rotation, self.energy], dtype=np.float32)}

    def evaluation_state(self):
        """Privileged labels; NEVER called by the learning data loader."""
        return np.array([*self.position/self.config.size, np.cos(self.angle),
                         np.sin(self.angle)], dtype=np.float32)

    def geometry(self):
        return {'rectangles': self.rectangles.tolist(), 'light': self.light.tolist(),
                'food': self.initial_food.tolist(), 'size': self.config.size}
