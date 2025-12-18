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

base_path = "../data/clean/irl"

all_dfs = []

for json_path in glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True):
    with open(json_path) as f:
        data = json.load(f)

    df_tmp = pd.json_normalize(data)
    df_tmp["source_file"] = json_path  # optionnel mais très utile pour debug

    all_dfs.append(df_tmp)

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
    X, Y, test_size=0.10, random_state=SEED, shuffle=True
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.1111111111, random_state=SEED, shuffle=True
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
            nn.Linear(6, 128),   # couche cachée
            nn.ReLU(),           # non-linéarité
            nn.Linear(128, 3)    # sortie
        )

    def forward(self, x):
        # Passage avant : simplement appliquer la couche linéaire
        return self.net(x)

# 2. Créer le modèle
model = SimpleNN()

# 3. Définir une fonction de perte et un optimiseur
criterion = nn.MSELoss()  # erreur quadratique moyenne
optimizer = optim.Adam(model.parameters(), lr=0.001)  # descente de gradient

# print(targets)

# 5. Entraînement minimal : une itération

epochs = 1300

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