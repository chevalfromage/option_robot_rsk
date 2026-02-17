import numpy as np

import torch
from .SimpleNN import SimpleNNMemory
import joblib
import numpy as np

from pathlib import Path
import json

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

RSK_NEURAL_SIMULATOR = Path(__file__).resolve().parent.parent
CLEAN = RSK_NEURAL_SIMULATOR / "data" / "clean"

def extract_orders_derivee_history(sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders = sample["orders"]
    derivee = sample["derivee"]
    derivee_history = sample["derivee_history"]

    orders_array = np.array([
        orders["dx"],
        orders["dy"],
        orders["dtheta"]
    ], dtype=float)

    derivee_array = np.array([
        derivee["x"],
        derivee["y"],
        derivee["theta_cos"],
        derivee["theta_sin"]
    ], dtype=float)

    history_array = np.array([
        [derivee["x"], derivee["y"], derivee["theta_cos"], derivee["theta_sin"]] for derivee in derivee_history
        ], dtype=float)
    return orders_array, derivee_array, history_array


def test_MLP():
    orders, derivee, history = extract_orders_derivee_history(data_in)
    if(len(history)!=0):
        history_flat = np.concatenate(list(history))
    else:
       history_flat = []

    model = SimpleNNMemory()
    model.load_state_dict(torch.load("rsk_neural_simulator/model/trained_model/simple_nn_test_anto.pth"))
    model.eval()

    x_scaler = joblib.load("rsk_neural_simulator/model/trained_model/x_scaler_test_anto.pkl")
    y_scaler = joblib.load("rsk_neural_simulator/model/trained_model/y_scaler_test_anto.pkl")


    x_input = np.concatenate([orders, derivee, history_flat]).reshape(1, -1)
    
    x_scaled = x_scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_scaled = model(x_tensor)
    y_scaled = y_scaled.cpu().numpy()
    prediction_velocity_robot = y_scaler.inverse_transform(y_scaled)[0]
    return prediction_velocity_robot

def graphique(resultat, resultat_attendu):
  fig, tab = plt.subplots(3,2)

  plot_resultat_x = [d["x"] for d in resultat]
  plot_resultat_attendu_x = [d["x"] for d in resultat_attendu]
  tab[0,0].plot(plot_resultat_x, label="resultat_x")
  tab[0,0].plot(plot_resultat_attendu_x, label="resultat_attendu_x")
  tab[0,0].legend()
  tab[0,0].set_xlabel("instants")
  tab[0,0].set_ylabel("vitesse [m/s]")
  tab[0,0].set_title("Axe X")

  plot_resultat_y = [d["y"] for d in resultat]
  plot_resultat_attendu_y = [d["y"] for d in resultat_attendu]
  tab[1,0].plot(plot_resultat_y, label="resultat_y")
  tab[1,0].plot(plot_resultat_attendu_y, label="resultat_attendu_y")
  tab[1,0].legend()
  tab[1,0].set_xlabel("instants")
  tab[1,0].set_ylabel("vitesse [m/s]")
  tab[1,0].set_title("Axe Y")

  plot_resultat_theta_cos = [d["theta_cos"] for d in resultat]
  plot_resultat_attendu_theta_cos = [d["theta_cos"] for d in resultat_attendu]
  tab[0,1].plot(plot_resultat_theta_cos, label="resultat_theta_cos")
  tab[0,1].plot(plot_resultat_attendu_theta_cos, label="resultat_attendu_theta_cos")
  tab[0,1].legend()
  tab[0,1].set_xlabel("instants")
  tab[0,1].set_ylabel("vitesse theta en cos")
  tab[0,1].set_title("Axe theta_cos")

  plot_resultat_theta_sin = [d["theta_sin"] for d in resultat]
  plot_resultat_attendu_theta_sin = [d["theta_sin"] for d in resultat_attendu]
  tab[1,1].plot(plot_resultat_theta_sin, label="resultat_theta_sin")
  tab[1,1].plot(plot_resultat_attendu_theta_sin, label="resultat_attendu_theta_sin")
  tab[1,1].legend()
  tab[1,1].set_xlabel("instants")
  tab[1,1].set_ylabel("vitesse theta en sin")
  tab[1,1].set_title("Axe theta_sin")

  plot_resultat_theta = [np.arctan2(d["theta_sin"],d["theta_cos"]) for d in resultat]
  plot_resultat_attendu_theta = [np.arctan2(d["theta_sin"],d["theta_cos"]) for d in resultat_attendu]
  tab[2,1].plot(plot_resultat_theta, label="resultat_theta")
  tab[2,1].plot(plot_resultat_attendu_theta, label="resultat_attendu_theta")
  tab[2,1].legend()
  tab[2,1].set_xlabel("instants")
  tab[2,1].set_ylabel("vitesse du theta ???")
  tab[2,1].set_title("Axe theta")

  plt.show()


if __name__ == "__main__":

  path_fichier_in = Path(CLEAN / "b2" / "cross.json")

  with path_fichier_in.open("r", encoding= "utf-8") as f:
     datas_fichier_in = json.load(f)

  resultat = []
  resultat_attendu = []

  for data_in in datas_fichier_in:
    resultat.append({"x" : test_MLP()[0], "y" : test_MLP()[1], "theta_cos" : test_MLP()[2], "theta_sin" : test_MLP()[3], "theta" : np.arctan2(test_MLP()[3],test_MLP()[2])})
    resultat_attendu.append(data_in['derivee_next'])
    # print(f"résultat : {resultat[-1]}, résultat attendu : {resultat_attendu[-1]}")

  graphique(resultat, resultat_attendu)
  