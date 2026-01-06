import numpy as np

import torch
from SimpleNN import SimpleNN

import joblib


import numpy as np

data_in = {
    "timestamp": 64.29150862499955,
    "path_id": 3,
    "path_name": "cross",
    "robot": "b1",
    "orders": {
      "dx": 0.3080870777403544,
      "dy": 1.2121434902361683,
      "dtheta": -2.8614291815589716
    },
    "derivee": {
      "x": 0.3742469974641705,
      "y": 0.2777188249319136,
      "theta": -1.3405165280914573
    },
    "derivee_next": {
      "x": 0.41247648051442637,
      "y": 0.21560242765957532,
      "theta": -2.825230081711586
    }
  }

def extract_orders_and_derivee(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    orders = sample["orders"]
    derivee = sample["derivee"]

    orders_array = np.array([
        orders["dx"],
        orders["dy"],
        orders["dtheta"]
    ], dtype=float)

    derivee_array = np.array([
        derivee["x"],
        derivee["y"],
        derivee["theta"]
    ], dtype=float)

    return orders_array, derivee_array


def test_MLP():
    orders, derivee = extract_orders_and_derivee(data_in)

    model = SimpleNN()
    model.load_state_dict(torch.load("rsk_neural_simulator/model/trained_model/simple_nn.pth"))
    model.eval()

    x_scaler = joblib.load("rsk_neural_simulator/model/trained_model/x_scaler.pkl")
    y_scaler = joblib.load("rsk_neural_simulator/model/trained_model/y_scaler.pkl")


    x_input = np.concatenate([orders, derivee]).reshape(1, -1)
    x_scaled = x_scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_scaled = model(x_tensor)
    y_scaled = y_scaled.cpu().numpy()
    prediction_velocity_robot = y_scaler.inverse_transform(y_scaled)[0]
    return prediction_velocity_robot

print(f"résultat : {test_MLP()}, résultat attendu : {data_in['derivee_next']}")