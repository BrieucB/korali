//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.


// General Simulation Options
std::string OPTIONS = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 2 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

/* OBSTACLES */

// TASK 0
std::string OBJECTShalfDisk = "halfDisk angle=10 xpos=0.6 bForced=1 bFixed=1 xvel=0.15 tAccel=5 radius=";

// TASK 1
std::string OBJECTSnaca = "NACA L=0.12 xpos=0.6 angle=0 fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=13.15 tAccel=5 Fpitch=";

// TASK 2 & SWARM cases
std::string OBJECTSstefanfish = "stefanfish T=1 xpos=0.6 bFixed=1 pid=1 L=";

// TASK 3
std::string OBJECTSwaterturbine = "waterturbine semiAxisX=0.05 semiAxisY=0.017 xpos=0.6 bForced=1 bFixed=1 xvel=0.2 angvel=-0.79 tAccel=0 ";

/* AGENTS */

// SINGLE AGENT
std::string AGENT = " \n\
stefanfish L=0.2 T=1";

std::string AGENTPOSX  = " xpos=";
std::string AGENTPOSY  = " ypos=";
std::string AGENTANGLE = " angle=";

// SWARMS
std::vector<std::vector<double>> initialPositions{{
	{0.90, 0.90},
	{0.90, 1.10},
	{1.20, 0.80},
	{1.20, 1.00},
	{1.20, 1.20},
	{1.50, 0.70},
	{1.50, 0.90},
	{1.50, 1.10},
	{1.50, 1.30},
	{1.80, 0.60},
	{1.80, 0.80},
	{1.80, 1.00},
	{1.80, 1.20},
	{1.80, 1.40},
	{2.10, 0.70},
	{2.10, 0.90},
	{2.10, 1.10},
	{2.10, 1.30},
	{2.40, 0.80},
	{2.40, 1.00},
	{2.40, 1.20},
	{2.70, 0.90},
	{2.70, 1.10},
	{3.00, 1.00},
}};