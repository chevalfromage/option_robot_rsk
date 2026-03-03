import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import glob
import joblib
import matplotlib.pyplot as plt
from typing import List
from .SimpleNN import SimpleNN, SimpleNN3, SimpleNNMemory, SimpleNN4
from rsk_neural_simulator.data.preparation_datas_positions import MEMORY_WINDOW, FUTUR_WINDOW

REPERE = 'R'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "clean"))
TRAINED_MODEL_DIR = os.path.join(SCRIPT_DIR, "trained_model")
TRAINED_MODEL_NAME = f"simple_nn_4{REPERE}.pth"
SCALER_X_NAME = f"x_scaler_4{REPERE}.pkl"
SCALER_Y_NAME = f"y_scaler_4{REPERE}.pkl"

print(TRAINED_MODEL_NAME)

all_dfs = []

json_pattern = os.path.join(base_path, "**", "*.json") # snake_in juste pour les tests sinon *

for json_path in glob.glob(json_pattern, recursive=True):
    if os.path.basename(json_path) == "cross.json" :
        continue 

    with open(json_path) as f:
        print(json_path)
        data = json.load(f)

    df_tmp = pd.json_normalize(data)
    df_tmp["source_file"] = json_path  # optionnel mais très utile pour debug

    all_dfs.append(df_tmp)


df = pd.concat(all_dfs, ignore_index=True)

print("Nombre total d'échantillons :", len(df))

# Si la colonne 'derivee_history' existe (MEMORY_WINDOW > 0 ), 
# l'exploser en colonnes
if f"history_{REPERE}" in df.columns:
    # Chaque entrée doit être une liste de dicts de longueur MEMORY_WINDOW
    for idx in range(MEMORY_WINDOW):
        # extraire les clés pour ce pas mémoire
        df[f"history_{REPERE}.{idx}.delta_t"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["delta_t"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "delta_t" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.x"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["x"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "x" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.y"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["y"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "y" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.theta"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["theta"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "theta" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.dx"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["dx"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dx" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.dy"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["dy"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dy" in h[idx] else 0.0
        )
        df[f"history_{REPERE}.{idx}.dtheta"] = df[f"history_{REPERE}"].apply(
            lambda h: h[idx]["dtheta"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dtheta" in h[idx] else 0.0
        )

    # Optionnel : enlever la colonne liste originale pour éviter confusion
    df = df.drop(columns=[c for c in [f"history_{REPERE}"] if c in df.columns])

if f"futur_{REPERE}" in df.columns:
    for idx in range(FUTUR_WINDOW):
        # extraire les clés pour ce pas mémoire
        df[f"futur_{REPERE}.{idx}.dtheta"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["dtheta"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dtheta" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.x"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["x"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "x" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.y"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["y"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "y" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.theta"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["theta"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "theta" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.dx"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["dx"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dx" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.dy"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["dy"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dy" in h[idx] else 0.0
        )
        df[f"futur_{REPERE}.{idx}.dtheta"] = df[f"futur_{REPERE}"].apply(
            lambda h: h[idx]["dtheta"] if isinstance(h, list) and len(h) > idx and isinstance(h[idx], dict) and "dtheta" in h[idx] else 0.0
        )

    # Optionnel : enlever la colonne liste originale pour éviter confusion
    df = df.drop(columns=[c for c in [f"futur_{REPERE}"] if c in df.columns])



# Ajouter les features mémoire (derivee_history t-1 .. t-MEMORY_WINDOW)
# pandas.json_normalize génère des colonnes nommées derivee_history.<idx>.<key>
mem_X_cols = []
for idx in range(MEMORY_WINDOW):
    mem_X_cols.extend(
        [
            f"history_{REPERE}.{idx}.delta_t",
            f"history_{REPERE}.{idx}.x",
            f"history_{REPERE}.{idx}.y",
            f"history_{REPERE}.{idx}.theta",
            f"history_{REPERE}.{idx}.dx",
            f"history_{REPERE}.{idx}.dy",
            f"history_{REPERE}.{idx}.dtheta",
        ]

    )
X_cols = mem_X_cols 


fut_Y_cols = []
for idx in range(FUTUR_WINDOW):
    fut_Y_cols.extend(
        [
            f"futur_{REPERE}.{idx}.delta_t",
            f"futur_{REPERE}.{idx}.x",
            f"futur_{REPERE}.{idx}.y",
            f"futur_{REPERE}.{idx}.theta",
            f"futur_{REPERE}.{idx}.dx",
            f"futur_{REPERE}.{idx}.dy",
            f"futur_{REPERE}.{idx}.dtheta",
        ]
    )
Y_cols = fut_Y_cols
print(f"Y_cols : {Y_cols}")

# Supprimer colonnes non-pertinentes si présentes
for col in ["path_name", "robot", "path_id", "timestamp"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Vérifier la présence des colonnes X_cols / Y_cols et créer les manquantes à zéro
missing_X = [c for c in X_cols if c not in df.columns]
missing_Y = [c for c in Y_cols if c not in df.columns]
if missing_X or missing_Y:
    print("Warning: colonnes manquantes détectées dans les données nettoyées.")
    if missing_X:
        print(f" - colonnes X manquantes ({len(missing_X)}):", missing_X[:10], "..." if len(missing_X)>10 else "")
        for c in missing_X:
            df[c] = 0.0
    if missing_Y:
        print(f" - colonnes Y manquantes ({len(missing_Y)}):", missing_Y[:10], "..." if len(missing_Y)>10 else "")
        for c in missing_Y:
            df[c] = 0.0

X = df[X_cols]
Y = df[Y_cols]

SEED = 42

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.4, random_state=SEED, shuffle=True
)

print(X_train.shape, X_val.shape, X_test.shape)

x_scaler = StandardScaler()
X_train_scaled  = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_val_scaled   = y_scaler.transform(Y_val)
Y_test_scaled  = y_scaler.transform(Y_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_scaled, dtype=torch.float32)

X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val_scaled, dtype=torch.float32)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test_scaled, dtype=torch.float32)


device = torch.device("cpu")
print("Using device:", device)

model = SimpleNN4().to(device)
X_train_t = X_train_t.to(device)
Y_train_t = Y_train_t.to(device)
X_val_t   = X_val_t.to(device)
Y_val_t   = Y_val_t.to(device)
X_test_t  = X_test_t.to(device)
Y_test_t  = Y_test_t.to(device)


criterion = nn.MSELoss() # fonction de loss
optimizer = optim.Adam(model.parameters(), lr=10e-4)  # descente de gradient

# print(targets)

# entrainement

epochs = 800
early_stop = 20
val_loss_prev = 0

# historique des loss pour les plotter 
train_loss_history: List[float] = []
val_loss_history: List[float] = []

# historique des loss par composantes pour identifier si une sortie dérive
output_labels = ["delta_t", "x", "y", "theta", "dx", "dy", "dtheta"]
train_output_history = [[] for _ in output_labels]
val_output_history = [[] for _ in output_labels]

for epoch in range(epochs):
    
    model.train()
    optimizer.zero_grad() 

    preds = model(X_train_t)
    # print(f"preds : {preds}")
    train_loss = criterion(preds, Y_train_t)  #calcul de la loss à cette epoch

    train_loss.backward() #backpropagation
    optimizer.step()

    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, Y_val_t)

    # stocker les MSE loss totales
    train_loss_history.append(train_loss.item())
    val_loss_history.append(val_loss.item())
    
    # calcul à la main des MSE par composantes
    train_output = torch.mean((preds - Y_train_t) ** 2, dim=0).tolist() 
    #==> [train_MSE_dx, train_MSE_dy, train_MSE_cos, train_MSE_sin]
    val_output = torch.mean((val_preds - Y_val_t) ** 2, dim=0).tolist() 
    #==> [val_MSE_dx, val_MSE_dy, val_MSE_cos, val_MSE_sin]
    
    # puis on les stock pour les plotter plus tard 
    for idx in range(len(output_labels)):
        train_output_history[idx].append(train_output[idx])
        val_output_history[idx].append(val_output[idx])
        
        
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} , "
            f"train MSE = {train_loss.item():.6f} , "
            f"val MSE = {val_loss.item():.6f}"
        )
        
    # test l'early stopping
    if(abs(val_loss_prev- val_loss.item())<=0.00001):
        counter_loss_stop+=1
    else:
        counter_loss_stop=0
    if(counter_loss_stop>=early_stop):
        print("early stopped at epoch :",epoch)
        break
    val_loss_prev = val_loss.item()

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t)
    test_loss = criterion(test_preds, Y_test_t)

print("Test MSE :", test_loss.item())

with torch.no_grad():
    train_preds_eval = model(X_train_t)
    train_loss_eval = criterion(train_preds_eval, Y_train_t)

print("Train (eval) MSE :", train_loss_eval.item())


os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# plot des loss totales
epochs_idx = range(1, len(train_loss_history) + 1)
plt.figure()
plt.plot(epochs_idx, train_loss_history, label="Train MSE")
plt.plot(epochs_idx, val_loss_history, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.xscale("log")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

#plot des loss par composantes
plt.figure()
for idx, label in enumerate(output_labels):
    plt.plot(epochs_idx, train_output_history[idx], label=f"train {label}")
    plt.plot(epochs_idx, val_output_history[idx], linestyle="--", label=f"val {label}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.xscale("log")
plt.title("Loss par composante de l'output")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()



torch.save(model.state_dict(), os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME))
joblib.dump(x_scaler, os.path.join(TRAINED_MODEL_DIR, SCALER_X_NAME))
joblib.dump(y_scaler, os.path.join(TRAINED_MODEL_DIR, SCALER_Y_NAME))
print("Modèle sauvegardé")

