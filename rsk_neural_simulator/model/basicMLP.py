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
from .SimpleNN import SimpleNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "clean"))
TRAINED_MODEL_DIR = os.path.join(SCRIPT_DIR, "trained_model")

all_dfs = []

json_pattern = os.path.join(base_path, "**", "*.json")
for json_path in glob.glob(json_pattern, recursive=True):
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


model = SimpleNN()


criterion = nn.MSELoss() # fonction de loss
optimizer = optim.Adam(model.parameters(), lr=10e-4)  # descente de gradient

# print(targets)

# entrainement

epochs = 1500
early_stop = 20
val_loss_prev =0

# historique des loss pour les plotter 
train_loss_history: List[float] = []
val_loss_history: List[float] = []

# historique des loss par composantes pour identifier si c'est theta qui marche pas
output_labels = ["x", "y", "theta"]
train_output_history = [[] for _ in output_labels]
val_output_history = [[] for _ in output_labels]

for epoch in range(epochs):
    
    model.train()
    optimizer.zero_grad() 

    preds = model(X_train_t) 
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
    #==> [train_MSE_x, train_MSE_y, train_MSE_theta]
    val_output = torch.mean((val_preds - Y_val_t) ** 2, dim=0).tolist() 
    #==> [val_MSE_x, val_MSE_y, val_MSE_theta]
    
    # puis on les stock pour les plotter plus tard 
    for idx in range(len(output_labels)):
        train_output_history[idx].append(train_output[idx])
        val_output_history[idx].append(val_output[idx])
        
        
    if epoch % 50 == 0:
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



torch.save(model.state_dict(), os.path.join(TRAINED_MODEL_DIR, "simple_nn.pth"))
joblib.dump(x_scaler, os.path.join(TRAINED_MODEL_DIR, "x_scaler.pkl"))
joblib.dump(y_scaler, os.path.join(TRAINED_MODEL_DIR, "y_scaler.pkl"))
print("Modèle sauvegardé")

