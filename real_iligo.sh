#!/bin/bash
#
# Injection/testing script.  Produce results for known grb beaming angle.

./grbeams_IDE.py delta,1.0 

./grbeams_IDE.py uniform 

./grbeams_IDE.py jeffreys 
