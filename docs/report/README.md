

# Report

## Specifications

The objective of the client was to develop a simulator as close as possible to reality for the robot-soccer-kit. A simulator was already operational, but it was based on basic physics and did not include complex notions such as drifting or robot imperfections. The final goal of this project is to enable advanced experiments, such as using reinforcement learning to improve team strategy on the football field.

Our project is divided into two main parts: first, the training of the model, and second, its integration into the simulator. The model was trained on diverse data representing the robot’s behavior, in particular its speed after each command. Then, the model was integrated into the simulator by replacing the original physics-based computation: at each step, the current speed and the command are given to the model, which predicts the next speed.

The expected output of this project is a simulator able to predict the future speed of the robot from its current state and control commands more accurately than the original simulator.

## Implemented approch

The first step was to build a suitable dataset for training. The robot was made to move in as many different situations as possible on the field. Data collection was done using the robot-soccer-kit API, which allowed us to easily record the robot states and commands. The model was then trained on this dataset.

Our model is implemented in PyTorch and is described in [SimpleNN](/src/rsk-neural-simulator/rsk_neural_simulator/model/SimpleNN.py). We started from a classical multilayer perceptron (MLP) architecture and adjusted its structure according to the number of inputs and outputs, while tuning it to reduce the training loss.

For the integration phase, we modified the original [simulator](/src/robot-soccer-kit/rsk/simulator.py) by removing the physics equations and replacing them with calls to our neural network, which now computes the robot’s motion.

## Analysis of results

![Texte alternatif](/assets/img/loss_exemple.png)


* **User tests**: Setup a methodology to test the efficiency of your project against users. It may use pre-experiment and post-experiment questionnaires. The most users the better to draw meaningful conclusions from your experiments. Radar diagrams are good to summarize such results.
* **Table of data**: Provide (short) extracts of data and relevant statistics (distribution, mean, standard deviation, p-values...)
* **Plots**: Most data are more demonstrative when represented as plots. 

Draw conclusions, **interpret** the results and make recommandations to your client for your future of the work.
It is totally fine to have results that are not as good as initially expected. Be honest and analyse why you did not manage to reach the objectives.
