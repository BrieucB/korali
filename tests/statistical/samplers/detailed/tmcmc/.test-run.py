#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "tmcmc-specifics.py"])
if r!=0:
  exit(r)

exit(0)
