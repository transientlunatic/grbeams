#!/bin/bash
#
# Injection/testing script.  Produce results for known grb beaming angle.

./grbeams_scenarios.py delta,0.5 \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 10.0
