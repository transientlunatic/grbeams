#!/bin/bash
#
# Injection/testing script.  Produce results for known grb beaming angle.

./grbeams_scenarios.py delta,1.0 

./grbeams_scenarios.py uniform 

./grbeams_scenarios.py jeffreys 
