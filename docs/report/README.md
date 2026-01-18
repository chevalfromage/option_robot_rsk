

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

To evaluate the efficiency of our approach, we mainly analyzed the evolution of the training loss. The loss curve shows a clear global decrease during training, which indicates that the model is learning the relationship between the robot state, the control commands, and the resulting speed.

However, the convergence is not perfectly smooth and some oscillations can be observed. This suggests that the model still has difficulties generalizing to all situations, in particular for abrupt changes of commands or high-speed motions. The final loss value remains higher than expected, which means that the prediction accuracy is not yet sufficient to fully replace a precise physical model.

During training, we monitored both the training loss and the validation loss. When the validation loss started to diverge from the training loss, this indicated the beginning of overfitting. At this point, the training was stopped in order to keep a model that generalizes better to unseen data rather than one that only fits the training set.

![Training exemple](/assets/img/loss_exemple.png)

To better understand the behavior of the model, the total loss was also decomposed into its different components (for instance, errors on each velocity axis). This component-wise analysis allows us to identify which parts of the motion are well predicted and which remain more problematic, and therefore provides a more detailed insight into the limitations of the current model. In our case the erros is mainly in the position insted of the angle.

![Training exemple](/assets/img/loss_component.png)

Once the model was trained, it was integrated into the simulator. The following figure shows a comparison between the actual robot motion and the motion predicted by the model, for each axis.

![Training exemple](/assets/img/actual_speed_MLP.jpg)

For now the predictions are fairly close, but the model tends to smooth out the motions, underestimating sharp changes or abrupt variations.