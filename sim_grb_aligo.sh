#!/bin/bash
#
# Injection/testing script.  Produce results for known grb beaming angle.

./grbeams_scenarios.py delta,0.5 \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0

./grbeams_scenarios.py uniform \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0

./grbeams_scenarios.py jeffreys \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0
