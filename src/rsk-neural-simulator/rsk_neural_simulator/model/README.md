# Model

This submodule contains the neural network architectures and the training pipeline used to learn robot motion dynamics from real-world datasets.

## Contents
- **SimpleNN.py**: definitions of MLP architectures (SimpleNN, SimpleNN3, SimpleNNMemory).
- **trainMLP.py**: training entry point (data preparation, scalers, training loop).
- **trained_model/**: output directory containing generated artifacts (model weights and scalers).

## Model Training

The model relies on the prepared datasets (see the data module) located at:
```bash
rsk_neural_simulator/data/clean/
```
From the repository root (with the virtual environment activated), run:
```bash
python -m rsk_neural_simulator.model.trainMLP
```

## Generated Artifacts

After training, the following files are saved to:
```bash
rsk_neural_simulator/model/trained_model/
```
- **simple_nn_memory.pth** :  trained model weights (PyTorch format)
- **x_scaler_memory.pkl** : scikit-learn input scaler
- **y_scaler_memory.pkl** : scikit-learn output scaler

The simulator depends on these files; they must be present and compatible (same MEMORY_WINDOW).

## Key Parameters

- **MEMORY_WINDOW** (in rsk_neural_simulator/data/preparation_datas.py): memory window size (affects input_dim).
- **EPOCHS** (in trainMLP.py): number of training epochs.
- **lr / optimizer**: learning rate and optimizer (defined in trainMLP.py).

## Practical Notes

- Check device detection (MPS / CUDA / CPU) if a GPU is available; the script attempts to use MPS automatically.
- If MEMORY_WINDOW is modified, regenerate the prepared datasets and retrain the model.

## Remarks

- The scalers (x_scaler, y_scaler) and the model must be consistent (same input dimensions). Errors such as "X has 7 features, but StandardScaler is expecting 67" indicate a mismatch between data preparation and model/scaler configuration. Regenerate the artifacts or adjust MEMORY_WINDOW.
- Keep copies of pretrained artifacts if reproducible configurations are required.
