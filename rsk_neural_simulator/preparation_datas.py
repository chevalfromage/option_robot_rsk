import json


datas_fichier = "irl/b1/cross.json"

# Lecture d'un fichier JSON avec encodage UTF-8
with open(datas_fichier, 'r', encoding='utf-8') as fichier:
    datas = json.load(fichier)

datas_out = []

for k in range(1,len(datas)):
    position_precedent = datas[k-1]["robot_pose"]
    position = datas[k]["robot_pose"]

    diff = {axe : (position[axe] - position_precedent[axe]) for axe in position}
    if(diff['x']!=0 and diff['y']!=0 and diff['theta']!=0):
        datas_out = {axe : (position[axe] - position_precedent[axe])/dt for axe in position}

        dt = datas[k]["timestamp"] - datas[k-1]["timestamp"]

        derivee = {axe : (position[axe] - position_precedent[axe])/dt for axe in position}
        if(derivee['x']!=0 and derivee['y']!=0 and derivee['theta']!=0):
            print(derivee)    
    
    datas.pop(k)
    
        

print(position.keys())