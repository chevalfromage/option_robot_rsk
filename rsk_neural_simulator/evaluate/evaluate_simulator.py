"""Replay un unique scénario RSK sur le simulateur et tracer XY + theta."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

import rsk
from rsk_neural_simulator.data.data_aquire import (
    DT,
    PARKING_POSES,
    RAW_ROOT,
    ROBOT_MAP,
    park_robot as data_park_robot,
)
from rsk_neural_simulator.data.paths import DEFAULT_PATHS, BasePath, _angle_wrap as data_angle_wrap

MIN_SLEEP = 1 / 120
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_RAW_ROOT = (DATA_DIR / RAW_ROOT).resolve()


def angle_wrap(angle: float) -> float:
    return float(data_angle_wrap(angle))


@dataclass
class Pose:
    x: float
    y: float
    theta: float

    def __post_init__(self) -> None:
        self.theta = angle_wrap(self.theta)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Pose"]:
        if not data:
            return None
        x, y, theta = data.get("x"), data.get("y"), data.get("theta")
        if any(value is None for value in (x, y, theta)):
            return None
        return cls(float(x), float(y), float(theta))


def pose_error(camera: Pose, simulator: Pose) -> Tuple[float, float]:
    pos_err = math.dist((camera.x, camera.y), (simulator.x, simulator.y))
    theta_err = abs(angle_wrap(camera.theta - simulator.theta))
    return pos_err, theta_err


def first_valid_pose(samples: Sequence[dict]) -> Optional[Pose]:
    for sample in samples:
        pose = Pose.from_dict(sample.get("robot_pose"))
        if pose:
            return pose
    return None


def snapshot_pose(robot) -> Optional[Pose]:
    pose = robot.pose
    if pose is None:
        return None
    return Pose(float(pose[0]), float(pose[1]), float(pose[2]))


def instantiate_path(path_name: str) -> BasePath:
    for path in DEFAULT_PATHS:
        if path.name == path_name:
            return copy.deepcopy(path)
    raise ValueError(f"Trajectoire inconnue: {path_name}")


def park_robot(client: rsk.Client, robot_key: str, settle: float = 0.5) -> None:
    if robot_key not in PARKING_POSES:
        return
    robot_attr = ROBOT_MAP.get(robot_key)
    if robot_attr is None:
        return
    robot = getattr(client, robot_attr)
    data_park_robot(robot, robot_key)
    time.sleep(settle)


def park_all_robots(client: rsk.Client) -> None:
    for robot_key in ROBOT_MAP:
        park_robot(client, robot_key, settle=0.1)


def replay_scenario(
    samples: Sequence[dict],
    robot_key: str,
    scenario_path: Path,
    client: rsk.Client,
) -> Tuple[List[Tuple[float, Optional[Pose], Optional[Pose]]], str]:
    robot_attr = ROBOT_MAP[robot_key]
    robot = getattr(client, robot_attr)
    path_name = samples[0].get("path_name", scenario_path.stem)
    path = instantiate_path(path_name)
    path.reset()

    initial = first_valid_pose(samples)
    if not initial:
        fallback = path.initial_pose()
        if fallback:
            initial = Pose(float(fallback[0]), float(fallback[1]), float(fallback[2]))

    last_sim_pose: Optional[Pose] = snapshot_pose(robot)
    if last_sim_pose is None and initial is not None:
        last_sim_pose = Pose(initial.x, initial.y, initial.theta)

    records: List[Tuple[float, Optional[Pose], Optional[Pose]]] = []
    start_ts = float(samples[0].get("timestamp", 0.0))
    prev_ts = start_ts
    path_finished = False

    for sample in samples:
        if not path_finished:
            target = path.current_target()
            _, orders = robot.goto_compute_order(target, skip_old=False)
            robot.control(*orders)

        ts = float(sample.get("timestamp", 0.0))
        sleep_dt = ts - prev_ts if prev_ts is not None else DT
        prev_ts = ts
        if not path_finished:
            time.sleep(max(MIN_SLEEP, sleep_dt if sleep_dt > 0 else DT))

        sim_pose = snapshot_pose(robot) if not path_finished else None
        if sim_pose:
            last_sim_pose = sim_pose
        elif not path_finished:
            sim_pose = last_sim_pose
        if not path_finished:
            finished = path.update((sim_pose.x, sim_pose.y, sim_pose.theta)) if sim_pose else path.update(None)
            if finished:
                path_finished = True

        rel_ts = ts - start_ts
        cam_pose = Pose.from_dict(sample.get("robot_pose"))
        records.append((rel_ts, cam_pose, sim_pose))

    robot.control(0.0, 0.0, 0.0)
    return records, path_name


def compute_rmse(records: Sequence[Tuple[float, Optional[Pose], Optional[Pose]]]) -> Tuple[float, float]:
    pos_errors: List[float] = []
    theta_errors: List[float] = []
    for _, cam_pose, sim_pose in records:
        if cam_pose and sim_pose:
            pos_err, theta_err = pose_error(cam_pose, sim_pose)
            pos_errors.append(pos_err)
            theta_errors.append(theta_err)

    if not pos_errors:
        return float("nan"), float("nan")

    def rmse(values: List[float]) -> float:
        return math.sqrt(sum(v * v for v in values) / len(values))

    return rmse(pos_errors), rmse(theta_errors)


def plot_results(
    records: Sequence[Tuple[float, Optional[Pose], Optional[Pose]]],
    robot_key: str,
    path_name: str,
    destination: Path,
    pos_rmse: float,
    theta_rmse: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_xy, ax_theta_cam, ax_theta_sim) = plt.subplots(1, 3, figsize=(15, 4.5))

    cam_xy = [(pose.x, pose.y) for _, pose, _ in records if pose]
    sim_xy = [(pose.x, pose.y) for _, _, pose in records if pose]
    if cam_xy:
        ax_xy.plot(*zip(*cam_xy), label="Camera", linewidth=2)
    if sim_xy:
        ax_xy.plot(*zip(*sim_xy), label="Simulateur", linewidth=2)
    ax_xy.set_title("Trajectoire XY")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend()

    cam_theta = [(ts, pose.theta) for ts, pose, _ in records if pose]
    sim_theta = [(ts, pose.theta) for ts, _, pose in records if pose]

    if cam_theta:
        ax_theta_cam.plot([ts for ts, _ in cam_theta], [th for _, th in cam_theta], label="Camera", linewidth=2)
    ax_theta_cam.set_title("Theta caméra")
    ax_theta_cam.set_xlabel("temps [s]")
    ax_theta_cam.set_ylabel("rad")
    ax_theta_cam.grid(True)
    ax_theta_cam.legend()

    if sim_theta:
        ax_theta_sim.plot([ts for ts, _ in sim_theta], [th for _, th in sim_theta], label="Simulateur", linewidth=2, color="tab:orange")
    ax_theta_sim.set_title("Theta simulateur")
    ax_theta_sim.set_xlabel("temps [s]")
    ax_theta_sim.set_ylabel("rad")
    ax_theta_sim.grid(True)
    ax_theta_sim.legend()

    text = f"RMSE position: {pos_rmse:.3f} m\nRMSE theta: {theta_rmse:.3f} rad"
    fig.suptitle(f"{robot_key} / {path_name}")
    fig.text(0.5, 0.02, text, ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def run_paths_for_robot(
    client: rsk.Client,
    robot_key: str,
    raw_root: Path,
    output_dir: Path,
) -> None:
    scenario_dir = raw_root / robot_key
    if not scenario_dir.exists():
        print(f"[WARN] Pas de dossier de données pour {robot_key} ({scenario_dir})")
        return

    print(f"=== Début robot {robot_key} ===")
    park_robot(client, robot_key)

    for path in DEFAULT_PATHS:
        scenario_path = scenario_dir / f"{path.name}.json"
        if not scenario_path.exists():
            print(f"[WARN] Fichier absent: {scenario_path}")
            continue
        samples = json.loads(scenario_path.read_text(encoding="utf-8"))
        if not samples:
            print(f"[WARN] Fichier vide: {scenario_path}")
            continue

        records, path_name = replay_scenario(samples, robot_key, scenario_path, client)
        pos_rmse, theta_rmse = compute_rmse(records)
        destination = output_dir / robot_key / f"{path_name}.png"
        plot_results(records, robot_key, path_name, destination, pos_rmse, theta_rmse)
        print(f"[OK] {robot_key}/{path_name} → {destination}")

    park_robot(client, robot_key)
    print(f"=== Fin robot {robot_key} ===")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Adresse IP du simulateur RSK")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Dossier où écrire les plots (par défaut: evaluate/results)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Dossier contenant les JSON enregistrés (par défaut: data/raw)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    raw_root = args.raw_root.resolve()
    if not raw_root.exists():
        raise SystemExit(f"Dossier de données introuvable: {raw_root}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with rsk.Client(host=args.host) as client:
        park_all_robots(client)
        for robot_key in ROBOT_MAP:
            run_paths_for_robot(client, robot_key, raw_root, output_dir)


if __name__ == "__main__":
    main()
