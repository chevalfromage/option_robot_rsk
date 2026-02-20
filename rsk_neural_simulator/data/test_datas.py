"""Ce fichier sert à tester les données nettoyés"""


import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import keyboard
import sys
from rsk import constants as rsk_constants

MAX_X = rsk_constants.field_length / 2
MAX_Y = rsk_constants.field_width / 2

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
CLEAN_ROOT = SCRIPT_DIR / "clean"
PLOTS_ROOT = SCRIPT_DIR / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DATASET_NAME = "random_waypoints" #mettre à none pour tout plotter
_PLOT_KEYS_DONE: set[str] = set()


path_cross = [
    (-MAX_X, -MAX_Y, np.pi / 4),
    (MAX_X, MAX_Y, -3 * (np.pi / 4)),
    (-MAX_X, MAX_Y, -np.pi / 4),
    (MAX_X, -MAX_Y, 3 * (np.pi / 4))]
x_pathW = [path_cross[k][0] for k in range(len(path_cross))]
y_pathW = [path_cross[k][1] for k in range(len(path_cross))]

fig, ax = plt.subplots(1,2)
ax[0].set_aspect('equal', adjustable='box')
pointsW = ax[0].scatter(0, 0, color='orange')
futur_pointsW = ax[0].scatter(0, 0, color='red')
pathW = ax[0].scatter(x_pathW, y_pathW)
flecheW = FancyArrowPatch(posA=(0,0), posB=(1,2), arrowstyle='->')
ax[0].set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax[0].add_patch(flecheW)

ax[1].set_aspect('equal', adjustable='box')
pointsR = ax[1].scatter(0, 0, color='orange')
futur_pointsR = ax[1].scatter(0, 0, color='red')
flecheR = FancyArrowPatch(posA=(0,0), posB=(1,2), arrowstyle='->')
ax[1].set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax[1].add_patch(flecheR)


plt.show(block=False)

def exit():
    keyboard.unhook_all()
    sys.exit()

def plot_data_terrain(data):
    global init_graph
    global points

    new_x_W = [data["history_W"][k]["x"] for k in range(len(data["history_W"]))]
    new_y_W = [data["history_W"][k]["y"] for k in range(len(data["history_W"]))]
    new_theta_W = [data["history_W"][k]["theta"] for k in range(len(data["history_W"]))]

    new_dx_W = data["history_W"][0]["dx"] 
    new_dy_W = data["history_W"][0]["dy"] 

    futur_x_W = [data["futur_W"][k]["x"] for k in range(len(data["futur_W"]))]
    futur_y_W = [data["futur_W"][k]["y"] for k in range(len(data["futur_W"]))]

    pointsW.set_offsets(np.c_[new_x_W, new_y_W])
    futur_pointsW.set_offsets(np.c_[futur_x_W, futur_y_W])
    flecheW.set_positions((new_x_W[0],new_y_W[0]), (new_dx_W, new_dy_W)) 
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_data_robot(data):
    global init_graph
    global points

    new_x_R = [data["history_R"][k]["x"] for k in range(len(data["history_R"]))]
    new_y_R = [data["history_R"][k]["y"] for k in range(len(data["history_R"]))]
    new_theta_R = [data["history_R"][k]["theta"] for k in range(len(data["history_R"]))]

    new_dx_R = data["history_R"][0]["dx"]
    new_dy_R = data["history_R"][0]["dy"]

    futur_x_R = [data["futur_R"][k]["x"] for k in range(len(data["futur_R"]))]
    futur_y_R = [data["futur_R"][k]["y"] for k in range(len(data["futur_R"]))]
    
    pointsR.set_offsets(np.c_[new_x_R, new_y_R])
    futur_pointsR.set_offsets(np.c_[futur_x_R, futur_y_R])
    flecheR.set_positions((new_x_R[0],new_y_R[0]), (new_dx_R, new_dy_R)) #/1.5 pour la visualisation
    fig.canvas.draw()
    fig.canvas.flush_events()

def show_datas(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    instant = 0
    while instant <= len(datas)-1:
        plot_data_terrain(datas[instant])
        plot_data_robot(datas[instant])
        
        key = keyboard.read_key()

        if key == 'droite':
            instant +=1
        elif key == 'gauche':
            instant -= 1

if __name__ == "__main__":

    keyboard.add_hotkey('esc', exit)

    for json_file in CLEAN_ROOT.rglob("b1/cross.json"):
        print(f"Traitement : {json_file}")
        show_datas(json_file)

    
