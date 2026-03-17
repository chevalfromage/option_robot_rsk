"""Ce fichier sert à préparer les données pour l'entraînement du modèle. (dérivées, lissage, etc...)"""

"""Les orders sont dans le repère robot. Les positions du robot dans le repère monde. On passe donc les orders dans le repère monde pour history_W et les positions dans le repère robot pour history_R"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# nombre d'instants precedents (en plus du current dt) à utiliser pour la prédiction
MEMORY_WINDOW = 2
FUTUR_WINDOW = 1

def passage_repere_robot(delta_t, liste_W, position_robot):

    new_x_W = [liste_W[k]["x"] for k in range(len(liste_W))]
    new_y_W = [liste_W[k]["y"] for k in range(len(liste_W))]
    new_x_step = [new_x_W[k] - position_robot["x"] for k in range(len(new_x_W))] #juste pour le calcul
    new_y_step = [new_y_W[k] - position_robot["y"] for k in range(len(new_y_W))] #juste pour le calcul
    new_theta_W = [liste_W[k]["theta"] for k in range(len(liste_W))]
    new_x_R = [new_x_step[k]*np.cos(position_robot["theta"]) + new_y_step[k]*np.sin(position_robot["theta"]) for k in range(len(new_x_W))] #positions passés dans le nouveau repere robot
    new_y_R = [new_x_step[k]*-np.sin(position_robot["theta"]) + new_y_step[k]*np.cos(position_robot["theta"]) for k in range(len(new_x_W))] #positions passés dans le nouveau repere robot
    new_theta_R = [new_theta_W[k] - position_robot["theta"] for k in range(len(new_theta_W))]

    new_dx_R = [(liste_W[k]["dx"]*np.cos(new_theta_R[k]) - liste_W[k]["dy"]*np.sin(new_theta_R[k]))/1.5 + new_x_R[k] for k in range(len(liste_W))]
    new_dy_R = [(liste_W[k]["dx"]*np.sin(new_theta_R[k]) + liste_W[k]["dy"]*np.cos(new_theta_R[k]))/1.5 + new_y_R[k] for k in range(len(liste_W))]
    new_dtheta_R = [liste_W[k]["dtheta"]/1.5 for k in range(len(liste_W))]
    
    history_R = [{"delta_t": delta_t[k], "x": new_x_R[k], "y": new_y_R[k], "theta": new_theta_R[k], "dx": new_dx_R[k], "dy": new_dy_R[k], "dtheta": new_dtheta_R[k]} for k in range(len(new_x_R))]
    return history_R

def passage_repere_monde(delta_t, data):

    new_x_W = [data[k]["x"] for k in range(len(data))]
    new_y_W = [data[k]["y"] for k in range(len(data))]
    new_theta_W = [data[k]["theta"] for k in range(len(data))]

    new_dx_W = [(np.cos(new_theta_W[k])*data[k]["dx"] - np.sin(new_theta_W[k])*data[k]["dy"])/1.5 + new_x_W[k] for k in range(len(data))] #/1.5 pour la visualisation
    new_dy_W = [(np.sin(new_theta_W[k])*data[k]["dx"] + np.cos(new_theta_W[k])*data[k]["dy"])/1.5 + new_y_W[k] for k in range(len(data))] #/1.5 pour la visualisation
    new_dtheta_W = [(new_theta_W[k] + data[k]["dtheta"]/1.5) for k in range(len(data))] #new_dtheta_W = [(new_theta_W[k] + data[k]["dtheta"]/1.5 -1.57) for k in range(len(data))]

    history_W = [{"delta_t": delta_t[k], "x": new_x_W[k], "y": new_y_W[k], "theta": new_theta_W[k], "dx": new_dx_W[k], "dy": new_dy_W[k], "dtheta": new_dtheta_W[k]} for k in range(len(new_x_W))]
    return history_W

def sort_timestamps(states, commands):
    indices_states = []
    indices_commands = []
    i, j =0, 0
    while i < len(states) and j <len(commands):
        if states[i]['timestamp']< commands[j]['timestamp']:
            indices_states.append(i+j)
            i +=1
        else:
            indices_commands.append(i+j)
            j +=1
            
def cleaner_data(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)
    datas_fichier_out = Path(f"out_{datas_fichier_in}")

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    datas_out = []

    #Supprimer données sans rafraîchissement de la cam
    print(datas['commands'])

    sort_timestamps(datas['states'], datas['commands'])

    for instant in range(len(datas['commands'])):
        command_prev = datas['commands'][instant-1]["orders"]
        command = datas[instant]["robot_pose"]

        diff = {axe: command[axe] - command_prev[axe] for axe in command}
        if diff["x"] != 0 and diff["y"] != 0 and diff["theta"] != 0:
            datas_out.append(datas[instant])

    theta_series = [entry["robot_pose"]["theta"] for entry in datas_out]
    x_series = [entry["robot_pose"]["x"] for entry in datas_out]
    y_series = [entry["robot_pose"]["y"] for entry in datas_out]

    theta_times = [entry["timestamp"] for entry in datas_out]

    orders = [datas_out[(k+1)%len(datas_out)]["orders"] for k in range(len(datas_out))]
    positions = [datas_out[k]["robot_pose"] for k in range(len(datas_out))]

    for i in range(len(datas_out)):
        datas_out[i]["robot_pose"]["x"] = x_series[i]
        datas_out[i]["robot_pose"]["y"] = y_series[i]
        datas_out[i]["robot_pose"]["theta"] = theta_series[i]


    zero = {"x": 0.0, "y": 0.0, "theta": 0.0, "dx": 0.0, "dy": 0.0, "dtheta": 0.0}

    for instant in range(len(datas_out)):
        pos = datas_out[instant]["robot_pose"]
        if(instant+1<len(datas_out)):
            dt = datas_out[instant+1]["timestamp"] - datas_out[instant]["timestamp"]
        else:
            dt = 0.03
        datas_out[instant]["delta_t"] = dt

        history_W = []
        for k in range(MEMORY_WINDOW):
            idx = instant - k
            if idx >= 0:
                history_W.append(dict({**positions[idx], **orders[idx]}))
            else:
                history_W.append(dict(zero))
        datas_out[instant]["history_W"] = history_W

    delta_t = [datas_out[k]["delta_t"] for k in range(len(datas_out))]

    for instant in range(len(datas_out)): #création de history_R
        delta_t_instant = [0.03]*len(delta_t)
        for k in range(MEMORY_WINDOW):
            if(instant-k>=0):
                delta_t_instant[k] = delta_t[instant-k]
        datas_out[instant]["history_R"] = passage_repere_robot(delta_t_instant, datas_out[instant]["history_W"], datas_out[instant]["history_W"][0])
    
    for instant in range(len(datas_out)): #passage des orders dans le repère monde
        delta_t_instant = [0.03]*len(delta_t)
        for k in range(MEMORY_WINDOW):
            if(instant-k>=0):
                delta_t_instant[k] = delta_t[instant-k]
        datas_out[instant]["history_W"] = passage_repere_monde(delta_t_instant, datas_out[instant]["history_W"])
    
    for instant in range(len(datas_out)): #ajout des états futur
        datas_out[instant]["futur_W"] = [zero]*FUTUR_WINDOW
        for instant_futur in range(FUTUR_WINDOW):
            if instant + instant_futur + 1 < len(datas_out):
                datas_out[instant]["futur_W"][instant_futur] = datas_out[instant + instant_futur + 1]["history_W"][0]
                # datas_out[instant]["futur_W"][instant_futur]['dtheta'] += 1.8
                # print(f"datas_out[instant + instant_futur + 1]['history_W'][0] : {datas_out[instant + instant_futur + 1]['history_W'][0]}")

        delta_t_instant = [0.03]*len(delta_t)
        for k in range(FUTUR_WINDOW):
            if(instant+k+1<len(delta_t)):
                delta_t_instant[k] = delta_t[instant+k+1]
        datas_out[instant]["futur_R"] = passage_repere_robot(delta_t_instant, datas_out[instant]["futur_W"], datas_out[instant]["history_W"][0])

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
    json_file = Path("rsk_neural_simulator\data\data.json")
    print(f"Traitement : {json_file}")
    cleaner_data(json_file)
