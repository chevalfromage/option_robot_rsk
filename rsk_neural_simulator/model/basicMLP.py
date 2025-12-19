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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "clean", "irl"))

all_dfs = []

json_pattern = os.path.join(base_path, "**", "*.json")
for json_path in glob.glob(json_pattern, recursive=True):
    with open(json_path) as f:
        data = json.load(f)

    df_tmp = pd.json_normalize(data)
    df_tmp["source_file"] = json_path  # optionnel mais très utile pour debug

    all_dfs.append(df_tmp)

if not all_dfs:
    raise FileNotFoundError(
        f"Aucun fichier JSON trouvé via le motif {json_pattern}.\n"
        "Vérifie que les données nettoyées sont présentes et que tu exécutes le script dans l'environnement attendu."
    )

df = pd.concat(all_dfs, ignore_index=True)

print("Nombre total d'échantillons :", len(df))


X_cols = [
    "orders.dx",
    "orders.dy",
    "orders.dtheta",
    "derivee.x",
    "derivee.y",
    "derivee.theta"
]

Y_cols = [
    "derivee_next.x",
    "derivee_next.y",
    "derivee_next.theta"
]

df = df.drop(columns=["path_name", "robot", "path_id", "timestamp"])

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

# 1. Définir le modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 3)  # couche de sortie
        )

    def forward(self, x):
        # Passage avant : simplement appliquer la couche linéaire
        return self.net(x)

# 2. Créer le modèle
model = SimpleNN()

# 3. Définir une fonction de perte et un optimiseur
criterion = nn.L1Loss()  # erreur quadratique moyenne
optimizer = optim.Adam(model.parameters(), lr=10e-4)  # descente de gradient

# print(targets)

# 5. Entraînement minimal : une itération

epochs = 500

for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    optimizer.zero_grad()

    preds = model(X_train_t)
    train_loss = criterion(preds, Y_train_t)

    train_loss.backward()
    optimizer.step()

    # --- VALIDATION ---
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, Y_val_t)

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"train MSE = {train_loss.item():.6f} | "
            f"val MSE = {val_loss.item():.6f}"
        )


model.eval()
with torch.no_grad():
    test_preds = model(X_test_t)
    test_loss = criterion(test_preds, Y_test_t)

print("Test MSE :", test_loss.item())

with torch.no_grad():
    train_preds_eval = model(X_train_t)
    train_loss_eval = criterion(train_preds_eval, Y_train_t)

print("Train (eval) MSE :", train_loss_eval.item())

# function to research of the best learning rate
#import matplotlib.pyplot as plt
#lrs = []
#losses = []
#for i in range(100):
#    lr = 1e-7 * (10 ** (i / 20))  # de 1e-7 à environ 1
#    lrs.append(lr)
#
#    optimizer = optim.Adam(model.parameters(), lr=lr)
#
#    model.train()
#    optimizer.zero_grad()
#
#    preds = model(X_train_t)
#    loss = criterion(preds, Y_train_t)
#
#    loss.backward()
#    optimizer.step()
#
#    losses.append(loss.item())
#    
#plt.plot(lrs, losses)
#plt.xscale('log')
#plt.xlabel("Learning Rate")
#plt.ylabel("Training Loss")
#plt.title("Learning Rate Finder")
#plt.grid(True)
#plt.show()
#plt.savefig("learning_rate_finder.png")
#
#print(f"Figure saved to learning_rate_finder.png")