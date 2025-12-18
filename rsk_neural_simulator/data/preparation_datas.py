import json
from pathlib import Path

def cleaner_data(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)
    datas_fichier_out = Path("clean") / datas_fichier_in.relative_to("raw")

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    datas_out = []

    #Supprimer données sans rafraîchissement
    for instant in range(1, len(datas)):
        pos_prev = datas[instant-1]["robot_pose"]
        pos = datas[instant]["robot_pose"]

        diff = {axe: pos[axe] - pos_prev[axe] for axe in pos}
        if diff["x"] != 0 and diff["y"] != 0 and diff["theta"] != 0:
            datas_out.append(datas[instant])

    # Ajouter la dérivée
    for instant in range(1, len(datas_out)):
        pos_prev = datas_out[instant-1]["robot_pose"]
        pos = datas_out[instant]["robot_pose"]
        dt = datas_out[instant]["timestamp"] - datas_out[instant-1]["timestamp"]

        derivee = {axe: (pos[axe] - pos_prev[axe]) / dt for axe in pos}
        datas_out[instant]["derivee"] = derivee

    # Dérivée à t+dt
    for instant in range(1, len(datas_out)-1):
        datas_out[instant]["derivee_next"] = datas_out[instant+1]["derivee"]

    # Nettoyage des clés
    keys_to_remove = ["ball_position", "robot_pose"]
    for d in datas_out:
        for k in keys_to_remove:
            d.pop(k, None)

    datas_out.pop(0)
    datas_out.pop(-1)

    # Écriture
    datas_fichier_out.parent.mkdir(parents=True, exist_ok=True)
    with open(datas_fichier_out, 'w', encoding='utf-8') as fichier:
        json.dump(datas_out, fichier, indent=2)


raw_root = Path("raw")

for json_file in raw_root.rglob("*.json"):
    print(f"Traitement : {json_file}")
    cleaner_data(json_file)
