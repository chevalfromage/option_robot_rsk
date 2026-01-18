# 📖 Developer documentation

## Purpose

This documentation is intended for future developers (e.g. students continuing the project) who want to understand the internal structure of the codebase and extend the neural simulation pipeline.

It focuses on **architecture**, **data flow**, and **extension points**, rather than usage instructions (covered in user-level documentation).

## Codebase overview

The project is organized around two main components:

- **`robot-soccer-kit/`**  
  Upstream RSK codebase, including the analytical simulator and game controller.

- **`rsk_neural_simulator/`**  
  Learning-based components used to model robot motion dynamics from real-world data.

The neural model is integrated inside the RSK simulator codepath (no change required at the controller/strategy level).


## Neural simulator architecture

The `rsk_neural_simulator` package is structured as follows:

rsk_neural_simulator/
├── data/
│ ├── clean/ # Cleaned trajectory datasets (JSON, generated)
│ └── preparation_datas.py # Feature extraction and temporal history handling
├── model/
│ ├── SimpleNN.py # Neural network architectures
│ ├── trainMLP.py # Training entry point
│ └── trained_model/ # Trained weights and scalers (generated artifacts)
├── evaluate/ # Evaluation and plotting utilities
└── init.py

- **`data/`**  
  Data acquisition, visualization, and preprocessing pipeline.  
  See: [`data/README.md`](../src/rsk-neural-simulator/rsk_neural_simulator/data)

- **`model/`**  
  Neural network definitions and training scripts.  
  See: [`model/README.md`](../src/rsk-neural-simulator/rsk_neural_simulator/model)

- **`evaluate/`**  
  Evaluation scripts and analysis utilities (offline, not used at runtime).

The training pipeline produces model weights and scalers that are loaded at runtime by the simulator.

## Integration with the RSK simulator

The neural model is used inside the RSK simulator as an alternative **velocity update model**.

Key integration points:
- Neural models are loaded in `rsk/simulator.py`
- Model selection is controlled via a single configuration flag:

  ROBOT_VELOCITY_MODEL = "original" | "mlp" | "trig" | "history"

* The simulator expects specific artifact filenames (model weights and scalers)

This design allows switching between analytical and neural dynamics without changing higher-level control or strategy code.

## Data flow summary

1. Real robot trajectories are collected using an external vision system
2. Data is cleaned and transformed into learning-ready features
3. A neural network is trained to predict next-step robot velocities
4. The trained model is loaded by the simulator at runtime

This pipeline is fully reproducible from raw data.

## Extending the project

Extension points include:

* Adding new input features or history representations (`data/`)
* Modifying neural architectures (`model/SimpleNN.py`)
* Training alternative models or loss functions (`model/trainMLP.py`)
* Introducing hybrid analytical / learned dynamics in the simulator

Developers are encouraged to keep the neural components **modular** and avoid hard dependencies on the simulator internals.

## Notes for maintainers

- Some datasets/models may currently be present in the repository; treat them as generated artifacts unless explicitly versioned on purpose.
- If artifacts are removed/cleaned, document the exact regeneration command in the relevant subpackage README.