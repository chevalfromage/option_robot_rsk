# RSK Neural Simulator

## Overview
This repository contains the learning-based components used to model **robot motion dynamics** in the Robot Soccer Kit (RSK) simulator.

Instead of relying solely on an analytical dynamics model, this package provides neural models trained from **real-world robot trajectories** acquired on physical RSK robots using an external vision system.  
The goal is to study data-driven dynamics modeling and reduce the sim-to-real gap within the existing RSK software stack.

This work is developed in an academic and pedagogical context and is intended as an experimental baseline rather than a production-ready simulator.

## Package structure
```bash
rsk_neural_simulator/
├── data/
│   ├── clean/                # Cleaned trajectory datasets (JSON)
│   └── preparation_datas.py  # Feature extraction and temporal history handling
├── model/
│   ├── SimpleNN.py           # Neural network architectures
│   ├── trainMLP.py           # Training script (entry point)
│   └── trained_model/        # Trained weights and scalers (generated)
├── evaluate/                 # Evaluation and plotting utilities
└── __init__.py
```
## Aquiring Training Data

To acquire raw data from the real-world robot soccer kit, please follow the [data module documentation](rsk_neural_simulator/data/README.md)

## Training the neural model

Training is performed using cleaned trajectory data located in:
```bash
rsk_neural_simulator/data/clean/
````

All `.json` files are discovered recursively.

From the **repository root**, with the virtual environment activated:

```bash
python -m rsk_neural_simulator.model.trainMLP
```

This script:

* Trains a multilayer perceptron (MLP) to predict robot velocity updates
* Automatically uses CUDA if available
* Generates the following artifacts at this path :

```bash
rsk_neural_simulator/model/trained_model/
```

* `simple_nn_memory.pth`
* `x_scaler_memory.pkl`
* `y_scaler_memory.pkl`

These filenames are **explicitly expected by the simulator code**.
If they are missing, the simulator will raise a `FileNotFoundError` at startup.

For more documentation about model training please check the [model documentation](rsk_neural_simulator/model/README.md).

## Integration with the RSK simulator

This package does **not** launch the simulator and does **not** expose runtime commands.

Its role is limited to:

* providing trained neural models,
* defining their architecture,
* and supplying the artifacts required by the RSK simulator.

The neural model is loaded by the simulator through direct imports and fixed file paths defined in `robot-soccer-kit/rsk/simulator.py`.
No dynamic configuration or plugin mechanism is used.

## Scope and limitations

* Only robot motion dynamics are learned (velocity update models)
* Ball dynamics remain analytical
* Robot–robot collisions are not learned
* Model quality depends strongly on dataset coverage and realism

This package is designed to be extended or modified for experimentation with alternative architectures, inputs, or training strategies.