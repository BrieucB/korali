#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

parser = argparse.ArgumentParser(prog='GFPT', description='Runs the GFPT algorithm on ABF2D')
parser.add_argument('--dir', help='directory of result files',  default='_result', required=False)
args = parser.parse_args()
setResultsDir(args.dir)
    
### Defining problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 60.0
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 1

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 60.0

### Defining state variables

e["Variables"][0]["Name"] = "Swimmer 1 - Pos X"
e["Variables"][1]["Name"] = "Swimmer 1 - Pos Y"
e["Variables"][2]["Name"] = "Swimmer 2 - Pos X"
e["Variables"][3]["Name"] = "Swimmer 2 - Pos Y"

### Defining action variables

e["Variables"][4]["Name"] = "Magnet Rotation X"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -1.0
e["Variables"][4]["Upper Bound"] = +1.0

e["Variables"][5]["Name"] = "Magnet Rotation Y"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -1.0
e["Variables"][5]["Upper Bound"] = +1.0

e["Variables"][6]["Name"] = "Magnet Intensity"
e["Variables"][6]["Type"] = "Action"
e["Variables"][6]["Lower Bound"] = +0.0
e["Variables"][6]["Upper Bound"] = +2.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / NAF"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Target Learning Rate"] = 0.001
e["Solver"]["Experiences Between Policy Updates"] = 20
e["Solver"]["Covariance Scaling"] = 0.01
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-6
e["Solver"]["Mini Batch Size"] = 32

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] =   20000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000
e["Solver"]["Mini Batch Strategy"] = "Prioritized"

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["File Output"]["Frequency"] = 100000

### Running Experiment

k.run(e)