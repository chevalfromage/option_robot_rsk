import threading
import time
import numpy as np
from numpy.linalg import norm
from math import dist
from . import kinematics, utils, constants, state, robot, robots, client

from collections.abc import Callable
from pathlib import Path

import sys
import os

from rsk_neural_simulator.data.preparation_datas import MEMORY_WINDOW
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Ensure sibling packages (e.g., rsk_neural_simulator) are importable
    sys.path.append(str(PROJECT_ROOT))
    

TRAINED_MODEL_DIR = PROJECT_ROOT / "rsk_neural_simulator" / "model" / "trained_model"
MODEL_PATH = TRAINED_MODEL_DIR / "simple_nn_test_anto.pth"
X_SCALER_PATH = TRAINED_MODEL_DIR / "x_scaler_test_anto.pkl"
Y_SCALER_PATH = TRAINED_MODEL_DIR / "y_scaler_test_anto.pkl"

import torch
from rsk_neural_simulator.model.SimpleNN import SimpleNN, SimpleNN3, SimpleNNMemory

import joblib
import warnings


# Constantes pour savoir quels robots faire spawn et ou 
SIMULATION_CONFIGURATION = "side"
# seulement les robots listés ici vont spawn dans la simulation
# mettre vide pour tous les faire spawn
#SIMULATION_MARKERS = {"green": {1}}
SIMULATION_MARKERS = {}


ROBOT_VELOCITY_MODEL = "history"  # "original", "mlp", "history" ou "trig"

class SimulatedObject:
    def __init__(
        self,
        marker: str,
        position: np.ndarray,
        radius: int,
        deceleration: float = 0,
        mass: float = 1,
    ) -> None:
        self.marker: str = marker
        self.radius: int = radius

        self.mass: float = mass
        self.position: np.ndarray = np.array([float(i) for i in position])

        self.velocity: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.deceleration: float = deceleration

        default_value = np.array([0.0, 0.0, 0.0, 0.0])
        self.velocity_history = deque([default_value.copy() for _ in range(MEMORY_WINDOW)], maxlen=MEMORY_WINDOW)

        self.pending_actions: list(Callable) = []
        self.sim: Simulator = None

    def execute_actions(self) -> None:
        for action in self.pending_actions:
            action()
        self.pending_actions = []

    def teleport(self, x: float, y: float, turn: float):
        self.position = np.array((x, y, turn))
        self.velocity = np.array([0.0, 0.0, 0.0])

    def update_velocity(self, dt) -> None:
        self.velocity[:2] = utils.update_limit_variation(
            self.velocity[:2], np.array([0.0, 0.0]), self.deceleration * dt
        )

    def collision_R(self, obj):
        """
        Given another object, computes the collision frame.
        It returns R_collision_world
        """

        # Computing unit vectors normal and tangent to contact (self to obj)
        normal = obj.position[:2] - self.position[:2]

        normal = (normal / norm(normal)) if norm(normal) != 0 else (0, 0)
        tangent = np.array([[0, -1], [1, 0]]) @ normal

        return np.vstack((normal, tangent))

    def collision(self, obj) -> None:
        R_collision_world = self.collision_R(obj)

        # Velocities expressed in the collision frame
        self_velocity_collision = R_collision_world @ self.velocity[:2]
        obj_velocity_collision = R_collision_world @ obj.velocity[:2]

        # Updating velocities using elastic collision
        u1 = self_velocity_collision[0]
        u2 = obj_velocity_collision[0]
        m1 = self.mass
        m2 = obj.mass
        Cr = 0.25

        # If the objects are not getting closer to each other, don't proceed
        if u2 - u1 > 0:
            return

        self_velocity_collision[0] = (m1 * u1 + m2 * u2 + m2 * Cr * (u2 - u1)) / (
            m1 + m2
        )
        obj_velocity_collision[0] = (m1 * u1 + m2 * u2 + m1 * Cr * (u1 - u2)) / (
            m1 + m2
        )

        # Velocities back in the world frame
        self.velocity[:2] = R_collision_world.T @ self_velocity_collision
        obj.velocity[:2] = R_collision_world.T @ obj_velocity_collision


class SimulatedRobot(SimulatedObject):
    def __init__(self, name: str, position: np.ndarray) -> None:
        super().__init__(
            name, position, constants.robot_radius, 0, constants.robot_mass
        )
        self.control_cmd: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.leds = None  # [R,G,B] (0-255) None for off

        missing_artifacts = [
            path for path in (MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH) if not path.exists()
        ]
        if missing_artifacts:
            missing_str = ", ".join(str(p) for p in missing_artifacts)
            raise FileNotFoundError(
                "Missing neural simulator artifacts: "
                f"{missing_str}. Run `python -m rsk_neural_simulator.model.basicMLP` "
                "from the project root to generate them."
            )

        self.model = SimpleNNMemory()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.x_scaler = joblib.load(X_SCALER_PATH)
        self.y_scaler = joblib.load(Y_SCALER_PATH)

        warnings.filterwarnings("ignore", category=UserWarning)

    def compute_kick(self, power: float) -> None:
        # Robot to ball vector, expressed in world
        ball_world = self.sim.objects["ball"].position[:2]
        T_world_robot = utils.frame(tuple(self.position))
        T_robot_world = utils.frame_inv(T_world_robot)
        ball_robot = utils.frame_transform(T_robot_world, ball_world)

        if utils.in_rectangle(
            ball_robot,
            [
                self.radius + constants.ball_radius - constants.kicker_x_tolerance,
                -constants.kicker_y_tolerance,
            ],
            [
                self.radius + constants.ball_radius + constants.kicker_x_tolerance,
                constants.kicker_y_tolerance,
            ],
        ):
            # TODO: Move the ball kicking velocity to constants
            # TODO: We should not set the ball velocity to 0 in the y direction but keep its velocity
            ball_speed_robot = [np.clip(power, 0, 1) * np.random.normal(0.8, 0.1), 0]
            self.sim.objects["ball"].velocity[:2] = (
                T_world_robot[:2, :2] @ ball_speed_robot
            )
            
    # fonction native du simulateur RSK qui utilise aucun MLP 
    def _update_velocity_original(self, dt: float) -> None:
        target_velocity_robot = self.control_cmd

        # mat de transformation du repère robot au repère monde
        T_world_robot = utils.frame(tuple(self.position))
        target_velocity_world = T_world_robot[:2, :2] @ target_velocity_robot[:2]

        # fait converger la vitesse actuelle vers la vitesse cible en limitant l'accélération
        self.velocity[:2] = utils.update_limit_variation(
            self.velocity[:2],
            target_velocity_world,
            constants.max_linear_acceleration * dt,
        )
        self.velocity[2:] = utils.update_limit_variation(
            self.velocity[2:],
            target_velocity_robot[2:],
            constants.max_angular_acceleration * dt,
        )

        #print(f"order : {self.control_cmd} , self.velocity : {self.velocity}")

    # fonction qu'on a ajouté pour compute next_speed,  via un MLP qui prend en entrée : 
    # ordre de vitesse + vitesse actuelle , et qui prédit la prochaine vitesse
    def _update_velocity_mlp(self, dt: float) -> None:
        target_velocity_robot = self.control_cmd 
        velocity_robot  = self.velocity

        prediction_velocity_robot: np.ndarray = np.array([0.0, 0.0, 0.0]) # sortie du MPL (vitesse prédite)

        # MLP ici
        #   entrées : target_velocity_robot (objectif de vitesse pour le robot), velocity_robot (vitesse actuelle du robot)
        #   sortie : prediction_velocity_robot

        x_input = np.concatenate([target_velocity_robot, velocity_robot]).reshape(1, -1)
        x_scaled = self.x_scaler.transform(x_input)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = self.model(x_tensor)
        y_scaled = y_scaled.cpu().numpy()
        prediction_velocity_robot = self.y_scaler.inverse_transform(y_scaled)[0]

        #fin du MLP

        T_world_robot = utils.frame(tuple(self.position))
        target_velocity_world = T_world_robot[:2, :2] @ prediction_velocity_robot[:2]

        self.velocity[:2] = target_velocity_world
        self.velocity[2:] = prediction_velocity_robot[2:]

        # print(f" marker : {self.marker} order : {self.control_cmd} , self.velocity : {self.velocity}")


    # fonction qui calcule la prochaine vitesse via un MLP qui utilise des fonctions trigonométriques pour éviter les 
    # discontinuités sur theta
    def _update_velocity_MLP_trig(self, dt: float) -> None:
        if dt <= 0:
            # Fallback to legacy behaviour if the simulator ever passes dt<=0
            self._update_velocity_original(max(dt, 0.0))
            return

        target_velocity_robot = self.control_cmd.astype(float)

        # Current velocity expressed in world. Convert to robot frame to match training features.
        T_world_robot = utils.frame(tuple(self.position))
        R_robot_world = T_world_robot[:2, :2]  # rotation from robot -> world
        velocity_world = self.velocity[:2]
        velocity_robot = R_robot_world.T @ velocity_world  # world -> robot

        theta = float(self.position[2])
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))

        nn_input = np.array(
            [
                target_velocity_robot[0],
                target_velocity_robot[1],
                target_velocity_robot[2],
                velocity_robot[0],
                velocity_robot[1],
                cos_theta,
                sin_theta,
            ]
        ).reshape(1, -1)

        x_scaled = self.x_scaler.transform(nn_input)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = self.model(x_tensor)
        prediction = self.y_scaler.inverse_transform(y_scaled.cpu().numpy())[0]

        vx_robot_next, vy_robot_next, cos_next, sin_next = prediction
        
        """ 
        # Normalise cos/sin in case of slight drift
        norm = float(np.hypot(cos_next, sin_next))
        if norm < 1e-6:
            cos_next, sin_next = cos_theta, sin_theta
            norm = 1.0
        cos_next /= norm
        sin_next /= norm  """

        theta_next = float(np.arctan2(sin_next, cos_next))
        dtheta = np.arctan2(np.sin(theta_next - theta), np.cos(theta_next - theta))
        omega_next = dtheta / dt

        R_next = np.array([[cos_next, -sin_next], [sin_next, cos_next]])
        velocity_world_next = R_next @ np.array([vx_robot_next, vy_robot_next])

        self.velocity[:2] = velocity_world_next
        self.velocity[2] = omega_next

    def _update_velocity_MLP_history(self, dt: float) -> None:
        if dt <= 0:
            # Fallback to legacy behaviour if the simulator ever passes dt<=0
            self._update_velocity_original(max(dt, 0.0))
            return

        target_velocity_robot = self.control_cmd.astype(float)

        # Current velocity expressed in world. Convert to robot frame to match training features.
        T_world_robot = utils.frame(tuple(self.position))
        R_robot_world = T_world_robot[:2, :2]  # rotation from robot -> world
        velocity_world = self.velocity[:2]
        velocity_robot = R_robot_world.T @ velocity_world  # world -> robot

        print(f"self.velocity: {self.velocity}")

        vtheta = float(self.velocity[2])
        vcos_theta = float(np.cos(vtheta))
        vsin_theta = float(np.sin(vtheta))
         
        history_flat = np.concatenate(list(self.velocity_history))

        nn_input = np.array(
            [
                target_velocity_robot[0],
                target_velocity_robot[1],
                target_velocity_robot[2],
                velocity_robot[0],
                velocity_robot[1],
                vcos_theta,
                vsin_theta,
                *history_flat
            ]
        ).reshape(1, -1)

        x_scaled = self.x_scaler.transform(nn_input)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = self.model(x_tensor)
        prediction = self.y_scaler.inverse_transform(y_scaled.cpu().numpy())[0]

        vx_robot_next, vy_robot_next, vcos_next, vsin_next = prediction

        print(f"target_velocity_robot : {target_velocity_robot}, velocity_robot : {velocity_robot} ,  prediction : {prediction}")
        
        """ 
        # Normalise cos/sin in case of slight drift
        norm = float(np.hypot(cos_next, sin_next))
        if norm < 1e-6:
            cos_next, sin_next = cos_theta, sin_theta
            norm = 1.0
        cos_next /= norm
        sin_next /= norm  """

        vtheta_next = float(np.arctan2(vsin_next, vcos_next))
        # dtheta = np.arctan2(np.sin(theta_next - theta), np.cos(theta_next - theta))
        # omega_next = dtheta / dt

        cos_next = np.cos(self.position[2]) + vcos_next*dt
        sin_next = np.sin(self.position[2]) + vsin_next*dt

        R_next = np.array([[cos_next, -sin_next], [sin_next, cos_next]])
        velocity_world_next = R_next @ np.array([vx_robot_next, vy_robot_next])

        self.velocity[:2] = velocity_world_next
        self.velocity[2] = vtheta_next

        print(f"new self.velocity : {self.velocity}")

        # mise à jour de l'historique glissant
        combined = np.array([velocity_robot[0], velocity_robot[1], vcos_theta, vsin_theta])
        self.velocity_history.appendleft(combined) 


    def update_velocity(self, dt: float) -> None:
        """Point d'enrtrée unique pour la MAJ des vitesses, selon le modèle choisi."""
        if ROBOT_VELOCITY_MODEL == "trig":
            self._update_velocity_MLP_trig(dt)
        elif ROBOT_VELOCITY_MODEL == "history":
            self._update_velocity_MLP_history(dt)
        elif ROBOT_VELOCITY_MODEL == "mlp":
            self._update_velocity_mlp(dt)
        elif ROBOT_VELOCITY_MODEL == "original":
            self._update_velocity_original(dt)
        else:
            raise ValueError(f"Modèle inconnu: {ROBOT_VELOCITY_MODEL}")


    def control_leds(self, r: int, g: int, b: int) -> None:
        self.leds = [r, g, b]



class RobotSim(robot.Robot):
    def __init__(self, url: str):
        super().__init__(url)
        self.set_marker(url)
        self.object: SimulatedRobot = None

    def initialize(self, position: np.ndarray) -> None:
        self.object = SimulatedRobot(self.marker, position)

    def teleport(self, x: float, y: float, turn: float) -> None:
        """
        Teleports the robot to a given position/orientation

        :param float x: x position [m]
        :param float y: y position [m]
        :param float turn: orientation [rad]
        """
        self.object.teleport(x, y, turn)

    def _control_original(self, dx: float, dy: float, dturn: float) -> None:
        self.object.control_cmd = kinematics.clip_target_order(
            np.array([dx, dy, dturn])
        )

    def _control_mlp(self, dx: float, dy: float, dturn: float) -> None:
        self.object.control_cmd = np.array([dx, dy, dturn])

    def _control_nn(self, dx: float, dy: float, dturn: float) -> None:
        """Prépare les commandes pour le NN afin de matcher le dataset d'entraînement."""
        order_world = np.array([dx, dy], dtype=float)
        T_world_robot = utils.frame(tuple(self.object.position))
        R_robot_world = T_world_robot[:2, :2]
        order_robot_xy = R_robot_world.T @ order_world

        order_robot = np.array([order_robot_xy[0], order_robot_xy[1], float(dturn)], dtype=float)
        self.object.control_cmd = kinematics.clip_target_order(order_robot) #order_robot

    def control(self, dx: float, dy: float, dturn: float) -> None:
        """Pareil que pour update_velocity, choisit la méthode control selon ROBOT_VELOCITY_MODEL."""
        if ROBOT_VELOCITY_MODEL == "trig":
            self._control_nn(dx, dy, dturn)
        elif ROBOT_VELOCITY_MODEL == "mlp":
            self._control_mlp(dx, dy, dturn)
        elif ROBOT_VELOCITY_MODEL == "history":
            self._control_mlp(dx, dy, dturn)
        elif ROBOT_VELOCITY_MODEL == "original":
            self._control_original(dx, dy, dturn)
        else:
            raise ValueError(f"Unknown ROBOT_VELOCITY_MODEL: {ROBOT_VELOCITY_MODEL}")

    def kick(self, power: float = 1.0) -> None:
        self.object.pending_actions.append(lambda: self.object.compute_kick(power))

    def leds(self, red: int, green: int, blue: int) -> None:
        """
        Controls the robot LEDs

        :param int red: red brightness (0-255)
        :param int green: green brightness (0-255)
        :param int blue: blue brightness (0-255)
        """
        self.object.pending_actions.append(
            lambda: self.object.control_leds(red, green, blue)
        )


class Simulator:
    def __init__(
        self, robots: robots.Robots, state: state.State = None, run_thread=True
    ):
        self.state: state.State = state
        self.robots: robots.Robots = robots

        # Creating the robots
        config_name = SIMULATION_CONFIGURATION
        if config_name not in client.configurations:
            raise KeyError(
                f"Unknown simulation configuration '{config_name}'. "
                f"Available presets: {', '.join(client.configurations.keys())}"
            )

        for configuration in client.configurations[config_name]:
            team, number = configuration[:2]
            if SIMULATION_MARKERS:
                allowed = SIMULATION_MARKERS.get(team)
                if allowed is None or number not in allowed:
                    continue
            robot: RobotSim = self.robots.add_robot(
                f"sim://{utils.robot_list2str(*configuration[:2])}"
            )
            robot.initialize(configuration[2])

        self.robots.update()

        self.objects: dict = {}

        # Creating the ball
        self.add_object(
            SimulatedObject(
                "ball",
                [0, 0, 0],
                constants.ball_radius,
                constants.ball_deceleration,
                constants.ball_mass,
            )
        )
        self.add_robot_objects()
        self.robots.ball = self.objects["ball"]

        self.fps_limit = 100

        if run_thread:
            self.run_thread()

    def run_thread(self):
        self.run = True
        self.simu_thread: threading.Thread = threading.Thread(
            target=lambda: self.thread(), daemon=True, name="SimulatorThread"
        )
        self.simu_thread.start()
        self.lock: threading.Lock = threading.Lock()

    def add_object(self, object: SimulatedObject) -> None:
        self.objects[object.marker] = object
        object.sim = self

    def add_robot_objects(self) -> None:
        for rob in self.robots.robots_by_marker.values():
            self.add_object(rob.object)

    def thread(self) -> None:
        last_time = time.time()
        while self.run:
            self.dt = -last_time + (last_time := time.time())
            self.loop(self.dt)

            while (self.fps_limit is not None) and (
                time.time() - last_time < 1 / self.fps_limit
            ):
                time.sleep(1e-3)

    def loop(self, dt):
        # Simulator proceed in two steps:
        # 1) We handle future collisions as elastic collisions and change the velocity vectors
        #    accordingly.
        # 2) We apply the object velocities, removing all the components in the velocities that would
        #    create collision.

        for obj in self.objects.values():
            # Execute actions (e.g: kick)

            # Update object velocity (e.g: deceleration, taking commands in account)
            obj.update_velocity(dt)

            if norm(obj.velocity) > 0:
                # Where the object would arrive without collisions
                future_pos = obj.position + obj.velocity * dt

                # Check for collisions
                for marker in self.objects:
                    if marker != obj.marker:
                        check_obj = self.objects[marker]
                        if dist(future_pos[:2], check_obj.position[:2]) < (
                            obj.radius + check_obj.radius
                        ):
                            obj.collision(check_obj)

        for obj in self.objects.values():
            # Check for collisions
            for marker in self.objects:
                if marker != obj.marker:
                    check_obj = self.objects[marker]
                    future_pos = obj.position + obj.velocity * dt

                    if dist(future_pos[:2], check_obj.position[:2]) < (
                        obj.radius + check_obj.radius
                    ):
                        R_collision_world = obj.collision_R(check_obj)
                        velocity_collision = R_collision_world @ obj.velocity[:2]
                        velocity_collision[0] = min(0, velocity_collision[0])
                        obj.velocity[:2] = R_collision_world.T @ velocity_collision

            obj.position = obj.position + (obj.velocity * dt)
            obj.execute_actions()
        if "ball" in self.objects and not utils.in_rectangle(
            self.objects["ball"].position[:2],
            [-constants.carpet_length / 2, -constants.carpet_width / 2],
            [constants.carpet_length / 2, constants.carpet_width / 2],
        ):
            self.objects["ball"].position[:3] = [0.0, 0.0, 0.0]
            self.objects["ball"].velocity[:3] = [0.0, 0.0, 0.0]

        self.push()

    def push(self) -> None:
        if self.state is not None:
            for marker in self.objects:
                pos = self.objects[marker].position
                vel = self.objects[marker].velocity
                if marker == "ball":
                    self.state.set_ball(pos[:2].tolist())
                    self.state.set_velocity("ball", vel[:2].tolist(), vel[2])
                    self.state.set_order("ball", None)
                else:
                    self.state.set_marker(marker, pos[:2].tolist(), pos[2])
                    self.state.set_velocity(marker, vel[:2].tolist(), vel[2])
                    self.state.set_leds(marker, self.objects[marker].leds)
                    control_cmd = getattr(self.objects[marker], "control_cmd", None)
                    if control_cmd is not None:
                        self.state.set_order(marker, control_cmd.tolist())
                    else:
                        self.state.set_order(marker, None)
