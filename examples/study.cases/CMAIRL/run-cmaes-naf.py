#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
sys.path.append('./_optimization_model')
from objfunc import *

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()
found = e.loadState('_korali_result_cmaes_naf/latest')
if (found == True):
  print('------------------------------------------------------')
  print('Continuing execution from latest...')
  print('------------------------------------------------------')
 
# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = rl_cartpole_naf

dim = 1

# Defining the problem's variables.
for i in range(dim):
    e["Variables"][i]["Name"] = "X" + str(i)
    e["Variables"][i]["Lower Bound"] = -1.0 # -57 degrees
    e["Variables"][i]["Upper Bound"] = +1.0 # +57 degrees

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 4
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 0.1
e["Solver"]["Termination Criteria"]["Max Generations"] = 500

# Configuring results path
e["File Output"]["Path"] = '_korali_result_cmaes_naf'
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1

e["Console Output"]["Verbosity"] = "Detailed"

k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 4

# Running Korali
k.run(e)