# Continuous_Control_RLND
This repository contains my submission Udacity's Deep Reinforcement Learning Nanodegree Project 2: [Continuous Control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)

## Environment
In this environment, a reward of +0.1 is provided of each step that a double-jointed arm is in a goal location. Thus, the objective of the agent is to maintain its position at the target location for as many time steps as possible.

<img width="500" height="251" alt="image" src="https://github.com/user-attachments/assets/e92f92e5-f3a3-49a4-b8c0-21618e03f403" />

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed training

The environment is based on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents), an open-siurce Unity plugin that enables games and simulations to serve as environments for training intelligent agents.
Udacity  provided two versions of the environment:
* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment.

In this submission I chose to work with the second version of the environment.

### Solving the environment
In order to solve the environment, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically:
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

## Getting Started
To run this project please configure a Python 3.6/PyTorch 0.4.0 environment with the requirements described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

You can download the already built environment according to your operating system here:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Finally, you can unzip the environment archive in the project's environment directory and set the path to the UnityEnvironment in the code.

