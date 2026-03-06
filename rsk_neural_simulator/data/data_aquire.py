"""
Collecte de données pour diverses trajectoires de robot RSK.
(trajectoires définies dans data/paths.py)
"""


import json
import math
import time
from pathlib import Path

import rsk
from rsk import constants as rsk_constants

from rsk_neural_simulator.data.paths import DEFAULT_PATHS, BasePath

# écart de temps entre chaque envoie d'ordres, 
# (calé sur le framerate théorique de la cam, mais il peut y avoir des drop)
# TODO : peut être se caler sur le framerate réel ?
DT = 1 / 30  #
RAW_ROOT = Path("raw")
# pour jouer la suite de path sur chaque robot.
ROBOT_MAP = {
    #"g1": "green1",
    #"g2": "green2",
    "b1": "blue1",
    #"b2": "blue2",
}
# poses ou iront se garer les robots après acquisition de données.
# Pour pas géner les autres robots, on se décale un peu en arrière.
PARKING_POSES = {
    "g1": (0.2, rsk_constants.field_width / 2 - 20, -math.pi / 2),
    "g2": (0.6, rsk_constants.field_width / 2 - 20, -math.pi / 2),
    "b1": (-0.2, rsk_constants.field_width / 2 - 20, -math.pi / 2),
    "b2": (-0.6, rsk_constants.field_width / 2 - 20, -math.pi / 2),
}

def record_paths_for_robot(client: rsk.Client, robot_key: str) -> None:
    """Enregistre les données de trajectoire pour un robot spécifique.  

    Args:
        client (rsk.Client): @IP du client RSK (controleur réél/ simulateur)
        robot_key (str): Clé identifiant le robot (à prendre dans robot map)
    """
    robot_attr = ROBOT_MAP[robot_key]
    robot = getattr(client, robot_attr)

    print(f"Acquisition de données pour {robot_key} ")
    run_start = time.monotonic()

    # loop sur chaque path défini dans paths.py
    for path_id, path in enumerate(DEFAULT_PATHS):

        path.reset()
        initial_pose = path.initial_pose()
       
        robot.goto(initial_pose)
        time.sleep(2)

        path_samples = []

        while True:
            # ici reprise du code d'origine de marc pour controler le robot 
            target_pose = path.current_target()
            _, orders = robot.goto_compute_order(target_pose)
            robot.control(*orders)

            pose = robot.pose if robot.pose is not None else (None, None, None)
            ball = client.ball if client.ball is not None else (None, None)

            #recup toutes les infos brutes pour ce dt
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

            if finished:
                break

        #construction du chemin ou stocker la data
        destination = RAW_ROOT / robot_key / f"{path.name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as f:
            json.dump(path_samples, f, indent=4)

        print(f"Data enregistrée dans {destination}")

    #retour parking
    park_robot(robot, robot_key)

def park_robot(robot, robot_key: str):
    '''Ramène le robot à sa position de parking après l'acquisition de données.
        (un peu en arrière pour éviter les collisions)
    Args: 
        robot : id du robot
        robot_key : g1/g2/g3/g4  
    '''
    pose = PARKING_POSES.get(robot_key)

    print(f"Retour parking pour {robot_key}")
    robot.goto(pose)
    robot.control(0, 0, 0)


def main(host: str = "192.168.100.1"):
    with rsk.Client(host=host) as client:
        # loop sur chaque robot
        for robot_key in ROBOT_MAP:
            try:
                record_paths_for_robot(client, robot_key)
            except Exception as exc:
                print(f"Acquisition ignorée pour {robot_key}: {exc}")


if __name__ == "__main__":
    main()