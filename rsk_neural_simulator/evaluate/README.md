# Simulator Evaluation Toolkit

This package replays the recorded trajectories collected with `data/data_aquire.py`
against the RSK simulator to quantify how close the physics model matches the
real-world robots. The workflow is:

1. Start the RSK simulator so it streams poses over ZMQ (typically by running the
   `robot-soccer-kit` project locally).
2. Run `evaluate_simulator.py` to stream the same velocity orders that were sent
to the real robots.
3. Compare the simulator poses with the camera-based ground truth. The script
   writes JSON comparisons, aggregated metrics, and XY/theta plots for each
   robot/path pair.

## Quick start

```bash
cd /Users/cesarlarragueta/Desktop/i3a/projet_RSK/option_robot_rsk
python -m rsk_neural_simulator.evaluate.evaluate_simulator \
  --host 127.0.0.1 \
  --dataset rsk_neural_simulator/data/raw \
  --output rsk_neural_simulator/evaluate/runs
```

Key options:

- `--host`: IP/hostname of the simulator controller.
- `--robots g1 g2`: restrict replay to a subset of robots.
- `--paths circle_in square`: restrict to specific trajectories.
- `--dt 0.025`: override the original sampling period.
- `--settle 1.5`: waiting time after teleporting before replaying commands.

Outputs are organised as follows:

```
runs/
  g1/
    circle_in.json          # per-sample camera vs simulator poses
  plots/
    g1/circle_in_xy.png     # XY overlay
    g1/circle_in_theta.png  # theta over time
  metrics_summary.json      # table of per-path RMSE/MAE values
```

## Requirements

The script relies on:

- `matplotlib`
- `numpy`
- Access to the `rsk` package (already included in `robot-soccer-kit`).

Install extra dependencies with:

```bash
pip install matplotlib numpy
```

## Tips

- Keep the simulator running throughout the replay session; the script reuses a
  single `rsk.Client` for all paths to minimize reconnection overhead.
- If the simulator drifts too far between paths, restart it or manually
  re-teleport robots before relaunching the script.
- You can run the script offline (without a simulator) to inspect saved metrics
  and plots; however, new comparisons require a reachable simulator endpoint.

## Minimal single-scenario helper

If you only need to replay one JSON log on the simulator and get a quick plot,
use `replay_single.py` (kept under 300 lines):

```bash
cd /Users/cesarlarragueta/Desktop/i3a/projet_RSK/option_robot_rsk
python -m rsk_neural_simulator.evaluate.replay_single \
  rsk_neural_simulator/data/raw/b1/circle_in.json \
  --host 127.0.0.1 \
  --output rsk_neural_simulator/evaluate/results
```

The script reads the selected log, drives the corresponding robot inside the
simulator, and saves a single PNG containing the XY trajectory and the theta vs
time plot, annotated with position/theta RMSE numbers.

Key options:

- `--host`: IP/hostname of the simulator controller.
- `--robots g1 g2`: restrict replay to a subset of robots.
- `--paths circle_in square`: restrict to specific trajectories.
- `--dt 0.025`: override the original sampling period.
- `--settle 1.5`: waiting time after teleporting before replaying commands.

Outputs are organised as follows:

```
results/
  b1_circle_in.png
```

## Dépendances

- `matplotlib`
- `rsk` (déjà présent dans `robot-soccer-kit`)

Installez `matplotlib` si besoin :

```bash
pip install matplotlib
```

## Conseils

- Laissez le simulateur tourner pendant toute la durée du scénario.
- Si le simulateur dérive, redémarrez-le ou re-téléportez le robot avant de
  relancer le script.
- Les timestamps du JSON sont utilisés pour l’axe temps, mais les ordres sont
  rejoués à 30 Hz fixes pour refléter l’acquisition d’origine.
# Simulator Evaluation Toolkit

This package replays the recorded trajectories collected with `data/data_aquire.py`
against the RSK simulator to quantify how close the physics model matches the
real-world robots. The workflow is:

1. Start the RSK simulator so it streams poses over ZMQ (typically by running the
   `robot-soccer-kit` project locally).
2. Run `evaluate_simulator.py` to stream the same velocity orders that were sent
to the real robots.
3. Compare the simulator poses with the camera-based ground truth. The script
   writes JSON comparisons, aggregated metrics, and XY/theta plots for each
   robot/path pair.

## Quick start

```bash
cd /Users/cesarlarragueta/Desktop/i3a/projet_RSK/option_robot_rsk
python -m rsk_neural_simulator.evaluate.evaluate_simulator \
  --host 127.0.0.1 \
  --dataset rsk_neural_simulator/data/raw \
  --output rsk_neural_simulator/evaluate/runs
```

### Minimal single-scenario helper

If you only need to replay one JSON log on the simulator and get a quick plot,
use `replay_single.py` (kept under 300 lines):

```bash
cd /Users/cesarlarragueta/Desktop/i3a/projet_RSK/option_robot_rsk
python -m rsk_neural_simulator.evaluate.replay_single \
  rsk_neural_simulator/data/raw/b1/circle_in.json \
  --host 127.0.0.1 \
  --output rsk_neural_simulator/evaluate/results
```

The script reads the selected log, drives the corresponding robot inside the
simulator, and saves a single PNG containing the XY trajectory and the theta vs
time plot, annotated with position/theta RMSE numbers.

Key options:

- `--host`: IP/hostname of the simulator controller.
- `--robots g1 g2`: restrict replay to a subset of robots.
- `--paths circle_in square`: restrict to specific trajectories.
- `--dt 0.025`: override the original sampling period.
- `--settle 1.5`: waiting time after teleporting before replaying commands.

Outputs are organised as follows:

```
runs/
  g1/
    circle_in.json          # per-sample camera vs simulator poses
  plots/
    g1/circle_in_xy.png     # XY overlay
    g1/circle_in_theta.png  # theta over time
  metrics_summary.json      # table of per-path RMSE/MAE values
```

## Requirements

The script relies on:

- `matplotlib`
- `numpy`
- Access to the `rsk` package (already included in `robot-soccer-kit`).

Install extra dependencies with:

```bash
pip install matplotlib numpy
```

## Tips

- Keep the simulator running throughout the replay session; the script reuses a
  single `rsk.Client` for all paths to minimize reconnection overhead.
- If the simulator drifts too far between paths, restart it or manually
  re-teleport robots before relaunching the script.
- You can run the script offline (without a simulator) to inspect saved metrics
  and plots; however, new comparisons require a reachable simulator endpoint.
