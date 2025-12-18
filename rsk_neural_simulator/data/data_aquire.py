"""
Collecte de données pour diverses trajectoires de robot RSK.

"""


import json
import math
import time
from pathlib import Path

import rsk
from rsk import constants as rsk_constants

from paths import DEFAULT_PATHS, BasePath

DT = 1 / 30  # 30 FPS
RAW_ROOT = Path("raw")
ROBOT_MAP = {
    "g1": "green1",
    "g2": "green2",
    "b1": "blue1",
    "b2": "blue2",
}
PARKING_POSES = {
    "g1": (0.2, rsk_constants.field_width / 2 + 20, -math.pi / 2),
    "g2": (0.6, rsk_constants.field_width / 2 + 20, -math.pi / 2),
    "b1": (-0.2, rsk_constants.field_width / 2 + 20, -math.pi / 2),
    "b2": (-0.6, rsk_constants.field_width / 2 + 20, -math.pi / 2),
}


def record_paths_for_robot(client: rsk.Client, robot_key: str) -> None:
    robot_attr = ROBOT_MAP[robot_key]
    robot = getattr(client, robot_attr)

    print(f"Acquisition pour {robot_key} ")
    run_start = time.monotonic()

    for path_id, path in enumerate(DEFAULT_PATHS):
        if not isinstance(path, BasePath):
            continue

        path.reset()
        initial_pose = path.initial_pose()
        if initial_pose is not None:
            robot.goto(initial_pose)
            time.sleep(2)

        path_samples = []
        last_tick = time.monotonic()
        while True:
            target_pose = path.current_target()
            _, orders = robot.goto_compute_order(target_pose)
            robot.control(*orders)

            pose = robot.pose if robot.pose is not None else (None, None, None)
            ball = client.ball if client.ball is not None else (None, None)
            path_samples.append(
                {
                    "timestamp": time.monotonic() - run_start,
                    "path_id": path_id,
                    "path_name": path.name,
                    "robot": robot_key,
                    "robot_pose": {"x": pose[0], "y": pose[1], "theta": pose[2]},
                    "ball_position": {"x": ball[0], "y": ball[1]},
                    "orders": {"dx": orders[0], "dy": orders[1], "dtheta": orders[2]},
                }
            )

            finished = path.update(robot.pose)

            last_tick += DT
            while time.monotonic() < last_tick:
                time.sleep(1e-3)

            if finished:
                break

        destination = RAW_ROOT / robot_key / f"{path.name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as f:
            json.dump(path_samples, f, indent=4)
        print(f"Data saved to {destination}")

    park_robot(robot, robot_key)


def park_robot(robot, robot_key: str) -> None:
    '''Returns the robot to its parking position after data acquisition.
        (un peu en arrière pour éviter les collisions)'''
    pose = PARKING_POSES.get(robot_key)
    if pose is None:
        return

    print(f"Retour parking pour {robot_key}")
    robot.goto(pose)
    robot.control(0, 0, 0)


with rsk.Client(host="127.0.0.1") as client:
    for robot_key in ROBOT_MAP:
        try:
            record_paths_for_robot(client, robot_key)
        except Exception as exc: 
            print(f"Acquisition ignorée pour {robot_key}: {exc}")