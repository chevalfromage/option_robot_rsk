"""Génère des plots XY/θ depuis les enregistrements JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from rsk_neural_simulator.data.data_aquire import RAW_ROOT, ROBOT_MAP
from rsk_neural_simulator.data.paths import DEFAULT_PATHS, wrap_angle

DATA_DIR = Path(__file__).resolve().parent
RAW_DEFAULT = (DATA_DIR / RAW_ROOT).resolve()
PLOT_DEFAULT = (DATA_DIR / "plots").resolve()


def parse_pose(pose): 
    x, y, theta = pose.get("x"), pose.get("y"), pose.get("theta")
    return float(x), float(y), wrap_angle(float(theta))


def load_samples(scenario_path):
    return json.loads(scenario_path.read_text(encoding="utf-8"))


#transforme les échantillons bruts en une liste de tuples (timestamp, Pose)
def extract_records(samples):
    records = []
    # timestamp de départ 
    start_ts = float(samples[0].get("timestamp", 0.0))
    
    for sample in samples:
        pose = parse_pose(sample.get("robot_pose"))
        ts = float(sample.get("timestamp", 0.0)) - start_ts
        records.append((ts, pose))
    return records


def plot_camera(records, robot_key, path_name, destination: Path): 

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_xy, ax_theta) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot XY
    xs = [pose[0] for _, pose in records]
    ys = [pose[1] for _, pose in records]
    ax_xy.plot(xs, ys, linewidth=2)
    ax_xy.set_title("Trajectoire XY")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    
    # Plot Theta
    times = [ts for ts, _ in records]
    thetas = [pose[2] for _, pose in records]
    ax_theta.plot(times, thetas, linewidth=2)
    ax_theta.set_title("Theta")
    ax_theta.set_xlabel("temps [s]")
    ax_theta.set_ylabel("rad")
    ax_theta.grid(True)
    
    fig.suptitle(f"{robot_key} / {path_name}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(destination)
    plt.close(fig)


def main():
    raw_root = RAW_DEFAULT
    output_root = PLOT_DEFAULT
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for robot_key in ROBOT_MAP:
        scenario_dir = raw_root / robot_key
        for path in DEFAULT_PATHS:
            scenario_path = scenario_dir / f"{path.name}.json"
            samples = load_samples(scenario_path)
            records = extract_records(samples)
            destination = output_root / robot_key / f"{path.name}.png"
            plot_camera(records, robot_key, path.name, destination)
            
if __name__ == "__main__":
    main()