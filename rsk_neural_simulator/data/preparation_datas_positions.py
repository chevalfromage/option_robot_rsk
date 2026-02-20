"""Ce fichier sert à préparer les données pour l'entraînement du modèle. (dérivées, lissage, etc...)"""

"""Les orders sont dans le repère robot. Les positions du robot dans le repère monde. On passe donc les orders dans le repère monde pour history_W et les positions dans le repère robot pour history_R"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

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
MEMORY_WINDOW = 10
FUTUR_WINDOW = 1

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

def passage_repere_robot(liste_W, position_robot):

    new_x_W = [liste_W[k]["x"] for k in range(len(liste_W))]
    new_y_W = [liste_W[k]["y"] for k in range(len(liste_W))]
    new_x_step = [new_x_W[k] - position_robot["x"] for k in range(len(new_x_W))] #juste pour le calcul
    new_y_step = [new_y_W[k] - position_robot["y"] for k in range(len(new_y_W))] #juste pour le calcul
    new_theta_W = [liste_W[k]["theta"] for k in range(len(liste_W))]
    new_x_R = [new_x_step[k]*np.cos(position_robot["theta"]) + new_y_step[k]*np.sin(position_robot["theta"]) for k in range(len(new_x_W))]
    new_y_R = [new_x_step[k]*-np.sin(position_robot["theta"]) + new_y_step[k]*np.cos(position_robot["theta"]) for k in range(len(new_x_W))]
    new_theta_R = [new_theta_W[k] - position_robot["theta"] for k in range(len(new_theta_W))]

    new_dx_R = [liste_W[k]["dx"]/1.5 for k in range(len(liste_W))]
    new_dy_R = [liste_W[k]["dy"]/1.5 for k in range(len(liste_W))]
    new_dtheta_R = [liste_W[k]["dtheta"]/1.5 for k in range(len(liste_W))]
    
    liste_R = [{"x": new_x_R[k], "y": new_y_R[k], "theta": new_theta_R[k], "dx": new_dx_R[k], "dy": new_dy_R[k], "dtheta": new_dtheta_R[k]} for k in range(len(new_x_R))]
    return liste_R

def passage_repere_monde(history_W):

    new_x_W = [history_W[k]["x"] for k in range(len(history_W))]
    new_y_W = [history_W[k]["y"] for k in range(len(history_W))]
    new_theta_W = [history_W[k]["theta"] for k in range(len(history_W))]

    new_dx_W = [(np.cos(new_theta_W[0])*history_W[k]["dx"] - np.sin(new_theta_W[0])*history_W[k]["dy"])/1.5 + new_x_W[0] for k in range(len(history_W))] #/1.5 pour la visualisation
    new_dy_W = [(np.sin(new_theta_W[0])*history_W[k]["dx"] + np.cos(new_theta_W[0])*history_W[k]["dy"])/1.5 + new_y_W[0] for k in range(len(history_W))] #/1.5 pour la visualisation
    new_dtheta_W = [(new_theta_W[k] + history_W[k]["dtheta"])/1.5 for k in range(len(history_W))]

    history_W = [{"x": new_x_W[k], "y": new_y_W[k], "theta": new_theta_W[k], "dx": new_dx_W[k], "dy": new_dy_W[k], "dtheta": new_dtheta_W[k]} for k in range(len(new_x_W))]
    return history_W

def cleaner_data(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)
    datas_fichier_out = CLEAN_ROOT / datas_fichier_in.relative_to(RAW_ROOT)

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    datas_out = []

    #Supprimer données sans rafraîchissement de la cam
    for instant in range(len(datas)):
        pos_prev = datas[instant-1]["robot_pose"]
        pos = datas[instant]["robot_pose"]

        diff = {axe: pos[axe] - pos_prev[axe] for axe in pos}
        if diff["x"] != 0 and diff["y"] != 0 and diff["theta"] != 0:
            datas_out.append(datas[instant])

    theta_series_raw = [entry["robot_pose"]["theta"] for entry in datas_out]
    #passage de x et y dans le repère robot
    x_series_raw = [entry["robot_pose"]["x"] for entry in datas_out]
    y_series_raw = [entry["robot_pose"]["y"] for entry in datas_out]
    # theta_series = smooth_series(theta_series_raw, THETA_SMOOTH_WINDOW, circular=True) # inutile ?
    # x_series = smooth_series(x_series_raw, POSITION_SMOOTH_WINDOW) # inutile ?
    # y_series = smooth_series(y_series_raw, POSITION_SMOOTH_WINDOW) # inutile ?
    theta_series = theta_series_raw
    x_series = x_series_raw
    y_series = y_series_raw
    theta_times = [entry["timestamp"] for entry in datas_out]

    orders = [entry["orders"] for entry in datas_out]
    positions = [entry["robot_pose"] for entry in datas_out]


    for i in range(len(datas_out)):
        datas_out[i]["robot_pose"]["x"] = x_series[i]
        datas_out[i]["robot_pose"]["y"] = y_series[i]
        datas_out[i]["robot_pose"]["theta"] = theta_series[i]

    zero = {"x": 0.0, "y": 0.0, "theta": 0.0, "dx": 0.0, "dy": 0.0, "dtheta": 0.0}

    for instant in range(len(datas_out)):
        pos = datas_out[instant]["robot_pose"]
        dt = datas_out[instant]["timestamp"] - datas_out[instant-1]["timestamp"]
        dt = 0.033

        history_W = []
        for k in range(MEMORY_WINDOW):
            idx = instant - k
            if idx >= 0:
                history_W.append(dict({**positions[idx], **orders[idx]}))
            else:
                history_W.append(dict(zero))
        datas_out[instant]["history_W"] = history_W

    for instant in range(len(datas_out)): #création de history_W
        datas_out[instant]["history_R"] = passage_repere_robot(datas_out[instant]["history_W"], datas_out[instant]["history_W"][0])
    
    for instant in range(len(datas_out)): #passage des orders dans le repère monde
        datas_out[instant]["history_W"] = passage_repere_monde(datas_out[instant]["history_W"])
    
    for instant in range(len(datas_out)): #ajout des états futur
        datas_out[instant]["futur_W"] = [zero]*FUTUR_WINDOW
        for instant_futur in range(FUTUR_WINDOW):
            if instant + instant_futur + 1 < len(datas_out):
                datas_out[instant]["futur_W"][instant_futur] = datas_out[instant + instant_futur + 1]["history_W"][0]
        
        datas_out[instant]["futur_R"] = passage_repere_robot(datas_out[instant]["futur_W"], datas_out[instant]["history_W"][0])

    # Nettoyage des clés
    keys_to_remove = ["ball_position", "robot_pose", "orders"]
    for d in datas_out:
        for k in keys_to_remove:
            d.pop(k, None)

    dataset_name = datas_fichier_out.stem

    # datas_out = [round_values(entry) for entry in datas_out]

    # Écriture
    datas_fichier_out.parent.mkdir(parents=True, exist_ok=True)
    with open(datas_fichier_out, 'w', encoding='utf-8') as fichier:
        json.dump(datas_out, fichier, indent=2)


if __name__ == "__main__":

    for json_file in RAW_ROOT.rglob("*.json"):
        print(f"Traitement : {json_file}")
        cleaner_data(json_file)
