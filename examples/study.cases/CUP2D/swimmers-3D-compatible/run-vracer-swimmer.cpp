#include "korali.hpp"
#if modelDIM == 2
#include "_model2D/swimmerEnvironment2D.hpp"
#else
#include "_model3D/swimmerEnvironment3D.hpp"
#endif


int main(int argc, char *argv[])
{
  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // retreiving number of task, agents, and ranks
  //int task    = atoi(argv[argc-5]);
  int nAgents = atoi(argv[argc-3]);
  int nRanks  = atoi(argv[argc-1]);

  // Storing parameters for environment
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  // Setting results path
  //std::string trainingResultsPath = "_trainingResults-"+std::to_string(modelDIM)+"D/";
  //std::string testingResultsPath = "_testingResults-"+std::to_string(modelDIM)+"D/";
  std::string trainingResultsPath = "_trainingResults/";
  std::string testingResultsPath = "_testingResults/";

  // Creating Korali experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";

  // Check if existing results are there and continuing them
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true){
    // printf("[Korali] Continuing execution from previous run...\n");
    // Hack to enable execution after Testing.
    e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
  }

  #if 0
  // Korali experiments for previous results
  auto eOld = korali::Experiment();
  auto found = eOld.loadState(trainingResultsPath + std::string("/latest"));
  if( found )
  {
    printf("[Korali] Continuing execution with policy learned in previous run...\n");
    e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = eOld["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"];
    e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
  }
  #endif

  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Agents Per Environment"] = nAgents;

  // Setting results path and dumping frequency in CUP
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

  // Setting up the state variables
  const size_t numStates = 7;

  size_t curVariable = 0;
  for (; curVariable < numStates; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("State") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  e["Variables"][curVariable]["Name"] = "Curvature";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -1.0;
  e["Variables"][curVariable]["Upper Bound"] = +1.0;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Swimming Period";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -0.25;
  e["Variables"][curVariable]["Upper Bound"] = +0.25;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Pitching Motion";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -20*0.2;
  e["Variables"][curVariable]["Upper Bound"] = +20*0.2;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 5*0.2;

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95;
  e["Solver"]["Mini Batch"]["Size"] =  128;

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

  //// Defining Neural Network
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";

  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria
  //e["Solver"]["Termination Criteria"]["Max Experiences"] = nAgents*5e5;
  e["Solver"]["Termination Criteria"]["Max Experiences"] = nAgents*5e7;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Normal";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Use Multiple Files"] = false;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  // Configuring conduit / communicator
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // ..and run
  k.run(e);
}
