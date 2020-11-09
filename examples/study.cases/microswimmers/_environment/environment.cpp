//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"

std::unique_ptr<msode::rl::MSodeEnvironment> _environment;
bool _isTraining;
std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];

  // Reseting environment and setting initial conditions
  _environment->reset(_randomGenerator, sampleId, true);

  // Setting initial state
  s["State"] = _environment->getState();

  // Calculating action scaling/shifting factors
  auto [lowerBounds, upperBounds] = _environment->getActionBounds();

  std::vector<double> scales = {
    (upperBounds[0] - lowerBounds[0]) * 0.5,
    (upperBounds[1] - lowerBounds[1]) * 0.5,
    (upperBounds[2] - lowerBounds[2]) * 0.5,
    (upperBounds[3] - lowerBounds[3]) * 0.5};

  std::vector<double> shifts = {
    (upperBounds[0] + lowerBounds[0]) * 0.5,
    (upperBounds[1] + lowerBounds[1]) * 0.5,
    (upperBounds[2] + lowerBounds[2]) * 0.5,
    (upperBounds[3] + lowerBounds[3]) * 0.5};

  // Defining status variable that tells us whether when the simulation is done
  Status status{Status::Running};

  // Storing action index
  size_t curActionIndex = 0;

  // Starting main environment loop
  while (status == Status::Running)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Scaling/shifting action
    for (size_t i = 0; i < action.size(); i++)
      action[i] = scales[i] * action[i] + shifts[i];

    // Printing Action:
    if (curActionIndex % 20 == 0)
    {
      printf("Action %lu: [ %f", curActionIndex, action[0]);
      for (size_t i = 1; i < action.size(); i++) printf(", %f", action[i]);
      printf("]\n");
    }

    // Setting action
    status = _environment->advance(action);

    // Storing reward
    s["Reward"] = _environment->getReward();

    // Storing new state
    s["State"] = _environment->getState();

    // Increasing action count
    curActionIndex++;
  }
}

void initializeEnvironment(const std::string confFileName)
{
  std::ifstream confFile(confFileName);

  if (!confFile.is_open())
    msode_die("Could not open the config file '%s'", confFileName.c_str());

  const Config config = json::parse(confFile);
  _environment = rl::factory::createEnvironment(config, ConfPointer(""));
}
