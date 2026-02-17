'''
Ce fichier définit une classe comme un Path à suivre pour le bot, 
que ce soit un waypointPath (lignes droites), ou un parametricPath 
(courbes définies par une fonction).
'''

import time
import random
from itertools import product
from typing import Callable, List, Optional, Sequence, Tuple, Union
from rsk import constants as rsk_constants
import numpy as np

# Fixer la seed pour la reproductibilité
np.random.seed(42)  
random.seed(42)     

MAX_X = rsk_constants.field_length / 2
MAX_Y = rsk_constants.field_width / 2

Pose = Tuple[float, float, float]

# TODO : eviter ca et faire plutot avec une erreur (visio marc 7/01)
def wrap_angle(value: float) -> float:
    """Calcule l'angle en wrappant entre -pi et pi évite les ovf"""
    return (value + np.pi) % (2 * np.pi) - np.pi


def build_grid_waypoints(
    x_step: float,
    y_step: float,
    theta_step: float,
    include_max: bool = True,
) -> List[Pose]:
    """Construit toutes les combinaisons (x, y, theta) de la grille et les mélange."""
    max_offset = x_step * 0.5 if include_max else 0.0
    xs = np.arange(-MAX_X, MAX_X + max_offset, x_step)
    ys = np.arange(-MAX_Y, MAX_Y + max_offset, y_step)
    thetas = np.arange(-np.pi, np.pi + (theta_step * 0.5 if include_max else 0.0), theta_step)

    waypoints = [(x, y, wrap_angle(theta)) for x, y, theta in product(xs, ys, thetas)]
    random.shuffle(waypoints)
    return waypoints



class BasePath:
    """Interface commune pour toutes les trajectoires."""

    def __init__(self, name: str):
        self.name = name
    
    # pour que ce soit une interface complete il faut 
    # définir les autres fonctions des classes qui l'implémente. 
    # elles seront implémentées dans les classes filles de cette interface.
    def reset(self) -> None:
        raise NotImplementedError

    def initial_pose(self) -> Optional[Pose]:
        raise NotImplementedError

    def current_target(self) -> Pose:
        raise NotImplementedError

    def update(self, current_pose: Optional[Sequence[float]]) -> bool:
        raise NotImplementedError

class WaypointPath(BasePath):
    """
    classe qui définit une trajectoire par une série de waypoints (lignes droites entre chaque point).
    """
    
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
        theta_error = abs(wrap_angle(pose_arr[2] - target_arr[2]))
        if pos_error <= self.tolerance and theta_error <= self.theta_tolerance:
            if self._index == len(self.waypoints) - 1:
                self._finished = True
            else:
                self._index += 1
        return self._finished

class PausingWaypointPath(WaypointPath):
    """
    Pareil que + haut mais avec des pauses à chaque waypoint.
    """

    def __init__(
        self,
        name: str,
        waypoints: Sequence[Pose],
        tolerance: float = 0.08,
        theta_tolerance: Optional[float] = None,
        pause_duration: Union[float, Tuple[float, float]] = 3.0,
    ) -> None:
        super().__init__(name, waypoints, tolerance=tolerance, theta_tolerance=theta_tolerance)
        self.pause_duration = pause_duration
        self._pause_until: Optional[float] = None
        self._current_pause = float(pause_duration) if isinstance(pause_duration, (int, float)) else None

    def reset(self) -> None:
        super().reset()
        self._pause_until = None

    def update(self, current_pose: Optional[Sequence[float]]) -> bool:
        # Si on est en phase de pause, on attend simplement
        if self._pause_until is not None:
            if time.monotonic() >= self._pause_until:
                # fin de la pause, on avance au waypoint suivant si possible
                if self._index < len(self.waypoints) - 1:
                    self._index += 1
                    self._pause_until = None
                    self._finished = False
                else:
                    self._finished = True
            return self._finished

        if current_pose is None:
            return self._finished

        pose_arr = np.array(current_pose)
        target_arr = np.array(self.waypoints[self._index])
        pos_error = np.linalg.norm(pose_arr[:2] - target_arr[:2])
        theta_error = abs(wrap_angle(pose_arr[2] - target_arr[2]))

        if pos_error <= self.tolerance and theta_error <= self.theta_tolerance:
            # on démarre une pause à ce waypoint
            self._pause_until = time.monotonic() + self._next_pause_duration()
            self._finished = False

        return self._finished

    def _next_pause_duration(self) -> float:
        """Retourne la durée de pause à appliquer (fixe ou aléatoire)."""
        if isinstance(self.pause_duration, tuple):
            low, high = self.pause_duration
            return random.uniform(low, high)
        return float(self.pause_duration)

"""
class ParametricPath(BasePath):

    classe qui définit les traj par une fonction paramétrée
    

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
"""

""" def _circle_pose(center: Tuple[float, float], radius: float, angular_speed: float, elapsed: float) -> Pose:
    angle = angular_speed * elapsed
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    #theta = np.arctan2(center[1] - y, center[0] - x) on garde le cercle toujours dans le même sens
    theta = 0.0
    
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
    return (x, y, orientation) """

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

""" # cercle en regardant à l'intérieur de la rotation
path5 = ParametricPath(
    "circle_in",
    pose_fn=lambda elapsed: _circle_pose((0.0, 0.0), MAX_Y, 0.5, elapsed),
    duration=20.0,
) """

""" # infini en regardant à l'exté des rotations
path6 = ParametricPath(
    "lemniscate_out",
    pose_fn=lambda elapsed: _lemniscate_pose(MAX_Y, 0.6, elapsed),
    duration=20.0,
)
 """
 
# Waypoints aléatoires
N_RANDOM_WAYPOINTS = 25
RANDOM_WAYPOINTS = [
    (random.uniform(-MAX_X, MAX_X), random.uniform(-MAX_Y, MAX_Y), random.uniform(-np.pi, np.pi)) for _ in range(N_RANDOM_WAYPOINTS)
]

path7 = PausingWaypointPath(
    "random_waypoints",
    RANDOM_WAYPOINTS,
    pause_duration=(3.0, 4.0),
)


# grille de points couvrant toutes les combinaisons possibles
GRID_WAYPOINTS = build_grid_waypoints(0.40, 0.40, np.pi / 8)

grid_path = PausingWaypointPath(
    "grid_cover",
    GRID_WAYPOINTS,
    pause_duration=(0, 3),
)


DEFAULT_PATHS: List[BasePath] = [
                                 
                                    grid_path
                                 ]



