#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from single_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

actions = [[i] for i in range(-20,20)]

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = actions
e["Problem"]["Environment Function"] = env
e["Problem"]["Action Repeat"] = 1
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Angle"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Cart Velocity"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Angulat Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Height Proxy"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Force"
e["Variables"][5]["Type"] = "Action"

### Configuring DQN hyperparameters

e["Solver"]["Type"] = "Agent / Discrete / DACER "
e["Solver"]["Importance Weight Truncation"] = 2.0
e["Solver"]["Optimization Steps Per Update"] = 1
e["Solver"]["Experiences Between Agent Trainings"] = 1
e["Solver"]["Experiences Between Target Network Updates"] = 1

e["Solver"]["Off Policy Updates"] = 12
e["Solver"]["Trajectory Size"] = 500

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 10000
e["Solver"]["Experience Replay"]["Maximum Size"] = 150000

e["Solver"]["Critic"]["Discount Factor"] = 0.99
e["Solver"]["Critic"]["Learning Rate"] = 0.01
e["Solver"]["Critic"]["Mini Batch Size"] = 32
e["Solver"]["Critic"]["Normalization Steps"] = 32

e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False

e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"
e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear"
e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = True

e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Weight Initialization Scaling"] = 0.001

## Defining Policy Configuration

e["Solver"]["Policy"]["Learning Rate"] = 0.01
e["Solver"]["Policy"]["Mini Batch Size"] = 32
e["Solver"]["Policy"]["Normalization Steps"] = 32

e["Solver"]["Policy"]["Trust Region"]["Enabled"] = True
e["Solver"]["Policy"]["Trust Region"]["Divergence Constraint"] = 0.5
e["Solver"]["Policy"]["Trust Region"]["Adoption Rate"] = 0.25

e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear"

e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 32
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh"

e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense"
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = False
e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Softmax" 


### Defining Termination Criteria

e["Solver"]["Training Reward Threshold"] = 750
e["Solver"]["Policy Testing Episodes"] = 20
e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 900

### Setting file output configuration

e["File Output"]["Frequency"] = 1000

### Running Experiment
k.run(e)