'''
This file defines a class as a Path for the bot to follow, 
whereas it's a waypointPath, or a parametricPath
'''

import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

MAX_X = 0.45
MAX_Y = 0.45

Pose = Tuple[float, float, float]


def _angle_wrap(value: float) -> float:
    """Calcule l'angle en wrappant entre -pi et pi évite les ovf"""
    return (value + np.pi) % (2 * np.pi) - np.pi


class BasePath:
    """Interface commune pour toutes les trajectoires."""

    def __init__(self, name: str):
        self.name = name
    
    # pour que ce soit une interface complete il faut 
    # définir les autres fonctions des classes qui l'implémente. 
    def reset(self) -> None:
        raise NotImplementedError

    def initial_pose(self) -> Optional[Pose]:
        raise NotImplementedError

    def current_target(self) -> Pose:
        raise NotImplementedError

    def update(self, current_pose: Optional[Sequence[float]]) -> bool:
        raise NotImplementedError


class WaypointPath(BasePath):
    def __init__(
        self,
        name: str,
        waypoints: Sequence[Pose],
        tolerance: float = 0.08,
        theta_tolerance: Optional[float] = None,
    ):
        super().__init__(name)
        if not waypoints:
            raise ValueError("WaypointPath requiert au moins un point")
        self.waypoints = list(waypoints)
        self.tolerance = tolerance
        self.theta_tolerance = theta_tolerance if theta_tolerance is not None else tolerance
        self._index = 0
        self._finished = False

    def reset(self) -> None:
        self._index = 0
        self._finished = False

    def initial_pose(self) -> Pose:
        return self.waypoints[0]

    def current_target(self) -> Pose:
        return self.waypoints[self._index]

    def update(self, current_pose: Optional[Sequence[float]]) -> bool:
        if current_pose is None:
            return self._finished

        pose_arr = np.array(current_pose)
        target_arr = np.array(self.waypoints[self._index])
        pos_error = np.linalg.norm(pose_arr[:2] - target_arr[:2])
        theta_error = abs(_angle_wrap(pose_arr[2] - target_arr[2]))

        if pos_error <= self.tolerance and theta_error <= self.theta_tolerance:
            if self._index == len(self.waypoints) - 1:
                self._finished = True
            else:
                self._index += 1
        return self._finished


class ParametricPath(BasePath):
    def __init__(
        self,
        name: str,
        pose_fn: Callable[[float], Pose],
        duration: Optional[float] = None,
    ):
        super().__init__(name)
        self.pose_fn = pose_fn
        self.duration = duration
        self._start_time = time.monotonic()

    def reset(self) -> None:
        self._start_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def initial_pose(self) -> Pose:
        return self.pose_fn(0.0)

    def current_target(self) -> Pose:
        return self.pose_fn(self._elapsed())

    def update(self, current_pose: Optional[Sequence[float]]) -> bool:
        if self.duration is None:
            return False
        return self._elapsed() >= self.duration


def _circle_pose(center: Tuple[float, float], radius: float, angular_speed: float, elapsed: float) -> Pose:
    angle = angular_speed * elapsed
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    theta = np.arctan2(center[1] - y, center[0] - x)
    return (x, y, theta)


def _lemniscate_pose(a: float, angular_speed: float, elapsed: float) -> Pose:
    theta = angular_speed * elapsed
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    denom = sin_t * sin_t + 1.0
    scale = a * np.sqrt(2.0) / denom
    x = scale * cos_t
    y = scale * sin_t * cos_t
    orientation = np.arctan2(y, x)
    return (x, y, orientation)


# carré en changeant l'angle à chaque waypoint
path1 = WaypointPath(
    "square",
    [
        (MAX_X, MAX_Y, 0.0),
        (MAX_X, -MAX_Y, np.pi / 2),
        (-MAX_X, -MAX_Y, np.pi),
        (-MAX_X, MAX_Y, -np.pi / 2),
    ],
)

# aller retours en largeur en regardant à l'extérieur
path2 = WaypointPath(
    "snake_out",
    [
        (MAX_X, MAX_Y, 0.0),
        (-MAX_X, 2 * (MAX_Y / 3), np.pi),
        (MAX_X, MAX_Y / 3, 0.0),
        (-MAX_X, 0, np.pi),
        (MAX_X, -MAX_Y / 3, 0.0),
        (-MAX_X, -2 * (MAX_Y / 3), np.pi),
        (MAX_X, -MAX_Y, 0.0),
    ],
)

# aller retours en hauteuru en regardant à l'inté
path3 = WaypointPath(
    "snake_in",
    [
        (MAX_X, -MAX_Y, np.pi / 2),
        (2 * (MAX_X / 3), MAX_Y, -np.pi / 2),
        ((MAX_X / 3), -MAX_Y, np.pi / 2),
        (0, MAX_Y, -np.pi / 2),
        (-(MAX_X / 3), -MAX_Y, np.pi / 2),
        (-2 * (MAX_X / 3), MAX_Y, -np.pi / 2),
        (-MAX_X, -MAX_Y, np.pi / 2),
    ],
)

# croix au milieu
path4 = WaypointPath(
    "cross",
    [
        (-MAX_X, -MAX_Y, np.pi / 4),
        (MAX_X, MAX_Y, -3 * (np.pi / 4)),
        (-MAX_X, MAX_Y, -np.pi / 4),
        (MAX_X, -MAX_Y, 3 * (np.pi / 4)),
    ],
)

# cercle en regardant à l'intérieur de la rotation
path5 = ParametricPath(
    "circle_in",
    pose_fn=lambda elapsed: _circle_pose((0.0, 0.0), MAX_Y, 0.5, elapsed),
    duration=20.0,
)

# infini en regardant à l'exté des rotations
path6 = ParametricPath(
    "lemniscate_out",
    pose_fn=lambda elapsed: _lemniscate_pose(MAX_Y, 0.6, elapsed),
    duration=20.0,
)


DEFAULT_PATHS: List[BasePath] = [path1, path2, path3, path4, path5, path6]



