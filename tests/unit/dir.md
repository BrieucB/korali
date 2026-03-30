# Directory Guide

## Purpose
Low-level Korali unit tests compiled and run through Meson/GTest.

## Specifics
This directory groups core unit-test entrypoints and module-focused subtests used to validate Korali behavior without running the full workflow stack.

## Provenance
Hand-maintained C++ unit tests for the local Korali fork.

## Immediate Contents
- Subdirectories: `modules`.
- Files: `engine.cpp`, `auxiliar.cpp`, `sample.cpp`, `meson.build`.

## Maintenance
Update this file whenever new unit-test areas are added or responsibilities shift.
