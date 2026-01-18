# Source code overview

This directory contains the source code of the project, organized into two main components that together form the simulation pipeline.

## robot-soccer-kit

This directory contains a local copy of the Robot Soccer Kit (RSK) software stack developed by the Rhoban laboratory.
(https://github.com/robot-soccer-kit/robot-soccer-kit)

It provides:
- the robot soccer kit

In this version, the simulator was replace by the developed neural simulator, while preserving the existing RSK interfaces and execution flow.

## rsk-neural-simulator

This directory contains the learning-based components introduced in this project.

It includes:
- data processing utilities for real-world robot trajectories,
- neural network models for robot motion dynamics,
- training and evaluation scripts,
- and pre-trained model weights used by the simulator.

The neural simulator is designed as a standalone Python package that need to be installed alongside the RSK codebase and imported by the simulator when needed.

## Relationship between components

The `robot-soccer-kit` component is responsible for simulation orchestration and execution, while `rsk-neural-simulator` provides the conception of the neural simulator.

The two components are loosely coupled: the neural simulator is loaded as an external dependency and can be enabled or disabled without modifying the overall RSK architecture.
