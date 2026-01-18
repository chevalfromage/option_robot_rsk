# Model

This module contains the neural network architectures and training pipeline used to learn robot motion dynamics from real-world data.

## Contents
- `SimpleNN.py`: neural network architectures (MLP variants)
- `trainMLP.py`: training entry point
- `trained_model/`: generated artifacts (model weights and scalers)

## Training the model

The model is trained from prepared datasets located in:

rsk_neural_simulator/data/clean/

From the **repository root** (with the virtual environment activated), run:
```bash
python -m rsk_neural_simulator.model.trainMLP
```

## Output artifacts

Training generates the following files in:

rsk_neural_simulator/model/trained_model/

* `simple_nn_memory.pth` – trained model weights
* `x_scaler_memory.pkl` – input feature scaler
* `y_scaler_memory.pkl` – output scaler

These files are required by the simulator to enable neural-based robot dynamics.