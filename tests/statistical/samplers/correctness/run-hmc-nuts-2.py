#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali

###### Static (No Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 1000
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Static'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Use Diagonal Metric"] = False
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71
e["Solver"]["Step Size Jitter"] = 0.001

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts-2a"

k = korali.Engine()
k.run(e)

# Testing Results
checkMean(e, 0.0, 0.1)
checkStd(e, 1.0, 0.3)

###### Static (Diagonal)

e = korali.Experiment()
e["Console Output"]["Frequency"] = 100
e["File Output"]["Enabled"] = False

e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 1000
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = True
e["Solver"]["Version"] = 'Static'
e["Solver"]["Use NUTS"] = True
e["Solver"]["Use Diagonal Metric"] = True
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Target Acceptance Rate"] = 0.71
e["Solver"]["Step Size Jitter"] = 0.001

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts-2b"

k = korali.Engine()
k.run(e)

# Testing Results
checkMean(e, 0.0, 0.1)
checkStd(e, 1.0, 0.3)
