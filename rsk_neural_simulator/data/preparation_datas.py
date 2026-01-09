"""Ce fichier sert à préparer les données pour l'entraînement du modèle. (dérivées, lissage, etc...)"""


import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
CLEAN_ROOT = SCRIPT_DIR / "clean"

# tentative d'arrondir tout au cm pour eviter les mouvbements brownien dans le simu
def round_values(value, ndigits=3):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: round_values(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_values(v, ndigits) for v in value]
    return value

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

    # on garde theta tel quel 
    theta_series = [entry["robot_pose"]["theta"] for entry in datas_out]

    # Lissage des positions x, y avant calcul des dérivées
    # i = mean(i, i-2, i-1, i+1, i+2)
    for i in range(len(datas_out)):
        start = max(0, i - 2)
        end = min(len(datas_out), i + 3) 
        window = datas_out[start:end]

        xs = [w["robot_pose"]["x"] for w in window]
        ys = [w["robot_pose"]["y"] for w in window]

        datas_out[i]["robot_pose"]["x"] = float(np.mean(xs))
        datas_out[i]["robot_pose"]["y"] = float(np.mean(ys))
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

    # Nettoyage des clés
    keys_to_remove = ["ball_position", "robot_pose"]
    for d in datas_out:
        for k in keys_to_remove:
            d.pop(k, None)

    # Suppression des valeurs extremes au cas où
    datas_out.pop(0)
    datas_out.pop(-1)

    datas_out = [round_values(entry) for entry in datas_out]

    # Écriture
    datas_fichier_out.parent.mkdir(parents=True, exist_ok=True)
    with open(datas_fichier_out, 'w', encoding='utf-8') as fichier:
        json.dump(datas_out, fichier, indent=2)


for json_file in RAW_ROOT.rglob("*.json"):
    print(f"Traitement : {json_file}")
    cleaner_data(json_file)
