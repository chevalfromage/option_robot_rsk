"""Génère des plots camera-only pour toutes les trajectoires enregistrées."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from rsk_neural_simulator.data.data_aquire import RAW_ROOT, ROBOT_MAP
from rsk_neural_simulator.data.paths import DEFAULT_PATHS

DATA_DIR = Path(__file__).resolve().parent
RAW_DEFAULT = (DATA_DIR / RAW_ROOT).resolve()
PLOT_DEFAULT = (DATA_DIR / "plots").resolve()


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose:
    x: float
    y: float
    theta: float

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Pose"]:
        if not data:
            return None
        x, y, theta = data.get("x"), data.get("y"), data.get("theta")
        if any(value is None for value in (x, y, theta)):
            return None
        return cls(float(x), float(y), angle_wrap(float(theta)))


def load_samples(scenario_path: Path) -> List[dict]:
    try:
        return json.loads(scenario_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[WARN] Fichier absent: {scenario_path}")
        return []


def extract_records(samples: Sequence[dict]) -> List[Tuple[float, Pose]]:
    records: List[Tuple[float, Pose]] = []
    if not samples:
        return records
    start_ts = float(samples[0].get("timestamp", 0.0))
    for sample in samples:
        pose = Pose.from_dict(sample.get("robot_pose"))
        if not pose:
            continue
        ts = float(sample.get("timestamp", 0.0)) - start_ts
        records.append((ts, pose))
    return records


def plot_camera(records: Sequence[Tuple[float, Pose]], robot_key: str, path_name: str, destination: Path) -> None:
    if not records:
        print(f"[WARN] Pas de données pour {robot_key}/{path_name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_xy, ax_theta) = plt.subplots(1, 2, figsize=(11, 4.5))

    xs = [pose.x for _, pose in records]
    ys = [pose.y for _, pose in records]
    ax_xy.plot(xs, ys, label="Camera", linewidth=2)
    ax_xy.set_title("Trajectoire XY")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend()

    times = [ts for ts, _ in records]
    thetas = [pose.theta for _, pose in records]
    ax_theta.plot(times, thetas, label="Camera", linewidth=2)
    ax_theta.set_title("Theta")
    ax_theta.set_xlabel("temps [s]")
    ax_theta.set_ylabel("rad")
    ax_theta.grid(True)
    ax_theta.legend()

    fig.suptitle(f"{robot_key} / {path_name}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    # argv conservé pour compat éventuelle, mais ignoré
    raw_root = RAW_DEFAULT
    output_root = PLOT_DEFAULT
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for robot_key in ROBOT_MAP:
        scenario_dir = raw_root / robot_key
        if not scenario_dir.exists():
            print(f"[WARN] Pas de dossier pour {robot_key} ({scenario_dir})")
            continue
        for path in DEFAULT_PATHS:
            scenario_path = scenario_dir / f"{path.name}.json"
            samples = load_samples(scenario_path)
            records = extract_records(samples)
            destination = output_root / robot_key / f"{path.name}.png"
            plot_camera(records, robot_key, path.name, destination)
if __name__ == "__main__":
    main()