
# Data

This module handles data acquisition, visualization, and preprocessing for training the neural dynamics models.

## Data acquisition
Collects real robot trajectories using an external vision system.

```bash
python -m rsk_neural_simulator.data.data_aquire
```

Make sure to be plugged into a RSK game controller with this @IP 

***192.168.100.1***

If the client is linked to the game controller, only the choosen robots in this dict will start their data recording. 
```python
# in data_aquire.py
ROBOT_MAP = {
    "g1": "green1",
    #"g2": "green2",
    "b1": "blue1",
    #"b2": "blue2",
}
```

They will start to follow the differents paths specified in ***paths.py*** in the variable **DEFAULT_PATHS**. 

After terminating each path, each robot will save it's raw data this way : 
```bash 
/data/raw/
        ├── b1
        │   ├── grid_cover.json
        │   ├── random_waypoints.json
        │   └── square.json
        └── g1
            ├── grid_cover.json
            ├── random_waypoints.json
            └── square.json
```

## Visualization

Plots recorded trajectories for inspection.

```bash
python -m rsk_neural_simulator.data.plot_traj
```

The plotted aquired trajectories can be found at this path 
```bash
/data/plots/
        ├── b1
        │   ├── cross.png
        │   ├── random_waypoints.png
        │   └── square.png
        └── g1
            ├── cross.png
            ├── random_waypoints.png
            └── square.png
```
## Data preparation

Cleans and formats data for neural network training.


In the current version of the project, the raw data are cleaned by the following steps (implemented in
`rsk_neural_simulator/data/preparation_datas.py`):

- Smooth orientation (theta) with a circular moving average using `THETA_SMOOTH_WINDOW`.
- Smooth positions X/Y with a moving average using `POSITION_SMOOTH_WINDOW`.
- Construct `derivee_history`: a fixed-length list of previous derivative dicts (t-1..t-MEMORY_WINDOW), padded with zeros if necessary.
- Round numeric values before saving (separate precision for positions and angles: `POSITION_ROUND` / `ANGLE_ROUND`).



Global parameters you can tweak (in `preparation_datas.py`):

- `THETA_SMOOTH_WINDOW` : window size for theta smoothing (larger = smoother orientation).
- `POSITION_SMOOTH_WINDOW` : window size for X/Y smoothing.
- `MEMORY_WINDOW` : number of previous timesteps stored in `derivee_history` (affects model input dimension).
- `POSITION_ROUND`, `ANGLE_ROUND` : decimal digits used when rounding positions and angles before saving.

To run the preparation step:

```bash
venv/bin/python -m rsk_neural_simulator.data.preparation_datas
```


Prepared datasets will be saved in:
```bash
rsk_neural_simulator/data/clean/
```

If you want to plot the comparison between your raw and smoothed data set this variable to the name of the path you want to plot : 

- `PLOT_DATASET_NAME` : limit debug plotting to a specific dataset (or `None` for all).

You will find the plots at this path :
```bash
data/plots/
        ├── smoothing_cross.png
        ├── smoothing_grid_cover.png
        └── smoothing_square.png
```

