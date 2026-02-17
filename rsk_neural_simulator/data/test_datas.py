"""Ce fichier sert à tester les données nettoyés"""


import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import keyboard
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_ROOT = SCRIPT_DIR / "raw"
CLEAN_ROOT = SCRIPT_DIR / "clean"
PLOTS_ROOT = SCRIPT_DIR / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DATASET_NAME = "random_waypoints" #mettre à none pour tout plotter
_PLOT_KEYS_DONE: set[str] = set()

x = 0
y = 0
fig, ax = plt.subplots()
points = ax.scatter(x, y)
fleche = FancyArrowPatch(posA=(0,0), posB=(1,2), arrowstyle='->')
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.add_patch(fleche)
plt.show(block=False)

def exit():
    keyboard.unhook_all()
    sys.exit()
    
def plot_data(data):
    global init_graph
    global points

    new_x = [data["history"][0]["x"] , data["history"][9]["x"]]
    new_y = [data["history"][0]["y"] , data["history"][9]["y"]]

    new_dx = data["history"][0]["dx"]
    new_dy = data["history"][0]["dy"]

    points.set_offsets(np.c_[new_x, new_y])
    fleche.set_positions((new_x[0],new_y[0]), (new_dx, new_dy))
    fig.canvas.draw()
    fig.canvas.flush_events()



def show_datas(datas_fichier_in):

    datas_fichier_in = Path(datas_fichier_in)

    # Lecture du JSON
    with open(datas_fichier_in, 'r', encoding='utf-8') as fichier:
        datas = json.load(fichier)

    instant = 0
    while instant <= len(datas):
        plot_data(datas[instant])
        
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

    
