#!/bin/bash
#
# Injection/testing script.  Produce results for known grb beaming angle.

../grbeams_ADE.py delta,0.5 \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0

../grbeams_ADE.py uniform \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0

../grbeams_ADE.py jeffreys \
    --sim-grbs \
    --sim-epsilon 0.5 --sim-theta 30.0
