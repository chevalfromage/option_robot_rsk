"""Ce fichier sert à préparer les données pour l'entraînement du modèle. (dérivées, lissage, etc...)"""


import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
CLEAN_ROOT = SCRIPT_DIR / "clean"
PLOTS_ROOT = SCRIPT_DIR / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DATASET_NAME = "random_waypoints" #mettre à none pour tout plotter
_PLOT_KEYS_DONE: set[str] = set()

THETA_SMOOTH_WINDOW = 15
POSITION_SMOOTH_WINDOW = 10

# nombre d'instants precedents (en plus du current dt) à utiliser pour la prédiction
MEMORY_WINDOW = 15

# tentative d'arrondir tout au cm pour eviter les mouvbements brownien dans le simu
def round_values(value, ndigits=3):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: round_values(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_values(v, ndigits) for v in value]
    return value

def smooth_series(values, window: int, circular: bool = False):
    """Lisse une série de valeurs avec une moyenne glissante."""
    if window <= 1:
        return [float(v) for v in values]
    half_window = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        window_vals = values[start:end]
        if circular:
            mean_cos = float(np.mean(np.cos(window_vals)))
            mean_sin = float(np.mean(np.sin(window_vals)))
            smoothed.append(float(np.arctan2(mean_sin, mean_cos)))
        else:
            smoothed.append(float(np.mean(window_vals)))
    return smoothed


def should_plot(dataset_name: str, plot_key: str):
    """Determine si on doit plotter ce dataset en fonction du filtre."""
    if PLOT_DATASET_NAME is not None and dataset_name != PLOT_DATASET_NAME:
        return False
    key = f"{dataset_name}:{plot_key}"
    if key in _PLOT_KEYS_DONE:
        return False
    _PLOT_KEYS_DONE.add(key)
    return True


def plot_smoothing_debug(times, raw_x, smooth_x, raw_y, smooth_y, raw_theta, smooth_theta, dataset_name: str):
    """Plot les résultats du lissage pour debug/ diapo"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(raw_x, raw_y, label="traj. brute", linewidth=1)
    axes[0].plot(smooth_x, smooth_y, label="traj. lissée", linewidth=1.2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Trajectoire XY")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    axes[1].plot(times, raw_theta, label="theta brut", linewidth=1)
    axes[1].plot(times, smooth_theta, label="theta lissé", linewidth=1.2)
    axes[1].set_xlabel("timestamp [s]")
    axes[1].set_ylabel("theta [rad]")
    axes[1].set_title("Theta vs temps")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    fig.suptitle(f"Smoothing debug - {dataset_name}")
    fig.tight_layout()
    out_path = PLOTS_ROOT / f"smoothing_{dataset_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot smoothing combiné sauvegardé: {out_path}")


def cleaner_data(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)
    datas_fichier_out = CLEAN_ROOT / datas_fichier_in.relative_to(RAW_ROOT)

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    datas_out = []

    #Supprimer données sans rafraîchissement de la cam
    for instant in range(1, len(datas)):
        pos_prev = datas[instant-1]["robot_pose"]
        pos = datas[instant]["robot_pose"]

        diff = {axe: pos[axe] - pos_prev[axe] for axe in pos}
        if diff["x"] != 0 and diff["y"] != 0 and diff["theta"] != 0:
            datas_out.append(datas[instant])

    theta_series_raw = [entry["robot_pose"]["theta"] for entry in datas_out]
    x_series_raw = [entry["robot_pose"]["x"] for entry in datas_out]
    y_series_raw = [entry["robot_pose"]["y"] for entry in datas_out]
    theta_series = smooth_series(theta_series_raw, THETA_SMOOTH_WINDOW, circular=True)
    x_series = smooth_series(x_series_raw, POSITION_SMOOTH_WINDOW)
    y_series = smooth_series(y_series_raw, POSITION_SMOOTH_WINDOW)
    theta_times = [entry["timestamp"] for entry in datas_out]

    for i in range(len(datas_out)):
        datas_out[i]["robot_pose"]["x"] = x_series[i]
        datas_out[i]["robot_pose"]["y"] = y_series[i]
        datas_out[i]["robot_pose"]["theta"] = theta_series[i]

    # Calculer dérivée de x, y et cos/sin theta 
    for instant in range(1, len(datas_out)):
        pos_prev = datas_out[instant-1]["robot_pose"]
        pos = datas_out[instant]["robot_pose"]
        dt = datas_out[instant]["timestamp"] - datas_out[instant-1]["timestamp"]

        derivee_x = (pos["x"] - pos_prev["x"]) / dt
        derivee_y = (pos["y"] - pos_prev["y"]) / dt

        theta_curr = theta_series[instant]

        derivee = {
            "x": derivee_x,
            "y": derivee_y,
            "theta_cos": float(np.cos(theta_curr)),
            "theta_sin": float(np.sin(theta_curr)),
        }

        datas_out[instant]["derivee"] = derivee

    # Valeurs à t+dt
    for instant in range(1, len(datas_out) - 1):
        datas_out[instant]["derivee_next"] = dict(datas_out[instant + 1]["derivee"])

    # Historique des dérivées (t-1, t-2, ..., t-MEMORY_WINDOW)
    # On crée pour chaque instant un champ `derivee_history` contenant une liste
    # de dictionnaires, l'entrée 0 correspondant à t-1, entrée 1 à t-2, etc.
    
    # on ajoute un historique des dérivées précédentes pour le modèle
    # sur une fenetre de mémoire de taille MEMORY_WINDOW
    # s'il manque les instants précédents, on remplit avec des zéros
    zero_derivee = {"x": 0.0, "y": 0.0, "theta_cos": 0.0, "theta_sin": 0.0}
    for instant in range(len(datas_out)):
        history = []
        for k in range(1, MEMORY_WINDOW + 1):
            idx = instant - k
            if idx >= 0:
                history.append(dict(datas_out[idx].get("derivee", zero_derivee)))
            else:
                history.append(dict(zero_derivee))
        datas_out[instant]["derivee_history"] = history

    # Nettoyage des clés
    keys_to_remove = ["ball_position", "robot_pose"]
    for d in datas_out:
        for k in keys_to_remove:
            d.pop(k, None)

    dataset_name = datas_fichier_out.stem
    plot_smoothing_debug(
        theta_times,
        x_series_raw,
        x_series,
        y_series_raw,
        y_series,
        theta_series_raw,
        theta_series,
        dataset_name,
    )

    # Suppression des valeurs extremes au cas où
    datas_out.pop(0)
    datas_out.pop(-1)

    datas_out = [round_values(entry) for entry in datas_out]

    # Écriture
    datas_fichier_out.parent.mkdir(parents=True, exist_ok=True)
    with open(datas_fichier_out, 'w', encoding='utf-8') as fichier:
        json.dump(datas_out, fichier, indent=2)


if __name__ == "__main__":
    for json_file in RAW_ROOT.rglob("*.json"):
        print(f"Traitement : {json_file}")
        cleaner_data(json_file)
