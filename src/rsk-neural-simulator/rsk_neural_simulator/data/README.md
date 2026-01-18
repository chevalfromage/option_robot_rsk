
# Data

This module handles data acquisition, visualization, and preprocessing for training the neural dynamics models.

## Data acquisition
Collects real robot trajectories using an external vision system.

```bash
python -m rsk_neural_simulator.data.data_aquire
```

## Visualization

Plots recorded trajectories for inspection.

```bash
python -m rsk_neural_simulator.data.plot_traj
```

## Data preparation

Cleans and formats data for neural network training.

```bash
python -m rsk_neural_simulator.data.preparation_datas
```

Prepared datasets are saved in:

rsk_neural_simulator/data/clean/