#!/bin/bash

python grb_beam.py --delta-eff-prior --eff-val 0.1
python grb_beam.py --delta-eff-prior --eff-val 0.5
python grb_beam.py --delta-eff-prior --eff-val 1.0
python grb_beam.py --flat-eff-prior --flat-eff-bounds=0.01,1
python grb_beam.py --log-eff-prior --log-eff-bounds=0.01,1
python grb_beam.py --berno-eff-prior
python grb_beam.py --beta-eff-prior --beta-vals 2,2
python grb_beam.py --beta-eff-prior --beta-vals 2,5
