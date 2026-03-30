# Directory Guide

## Purpose
Korali implementation of the `Hierarchical/Psi` problem used for Phase 2 hierarchical inference.

## Specifics
This directory contains the `Psi` C++ implementation plus the matching generated-source inputs (`psi.cpp.base`, `psi.hpp.base`, `psi.config`). The module loads completed sub-experiment posteriors, updates conditional priors from hyperparameters, and evaluates hierarchical log-likelihoods.

## Provenance
Hand-maintained Korali module code with generated-code templates committed in-tree.

## Immediate Contents
- Files: `psi.cpp`, `psi.hpp`, `psi.cpp.base`, `psi.hpp.base`, `psi.config`, `meson.build`, `README.rst`.

## Maintenance
Keep the emitted sources and their template/config inputs aligned whenever this module changes.
