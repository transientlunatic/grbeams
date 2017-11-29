#!/usr/bin/env python
# Copyright (C) 2016-2017 Daniel Williams <daniel.williams@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""This script is designed to reproduce the surface plots shown in
figures 7 and 8 of Williams et al. 2017.
"""

import argparse

parser = argparse.ArgumentParser(description='Produce a set of upper and lower bounds on the beaming angle for a range of horizon distances and BNS detections')
parser.add_argument('hor_lower',
                    metavar='start',
                    type=float,
                    help='The lowest horizon distance')
parser.add_argument('--prior', dest='prior',
                    metavar='efficiency_prior',
                    type=str,
                    default='jeffreys',
                    help='The prior distribution of efficiency')
args = parser.parse_args()

import pymc3 as pm
import numpy as np
import theano

hor_start = args.hor_lower
eff_prior = args.prior


from theano.compile.ops import as_op
import matplotlib
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)



def background_rate_f(b, T, n):
    """
    
    """
    out = 0
    #n = int(n)
    for i in range(n+1):
        out += ((b*T)**i * np.exp(- b*T)) / np.math.factorial(i)
    return out

def log_background_rate(b, T, n):
    return np.log(background_rate_f(b, T, n))

def signal_rate_part(s, n, b, T):
    top_a = T * ((s + b) * T)**n 
    top_b = np.exp(-(s + b)*T)
    p = (top_a * top_b) / np.math.factorial(n)
    return theano.tensor.switch(theano.tensor.le(s, 0), 0, p)

#@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dscalar])
def log_signal_rate(s,n,b,T):
    #if theano.tensor.lt(0, s): return np.array([[0.0]])
    p = -log_background_rate(b,T,n) + np.log(signal_rate_part(s,n,b,T))
    return p

def number_mweg(volume):
    """
    Calculates the number of MWEGs in a volume, given in units of Mpc^3
    """
    return volume * (0.0116) 

import theano.tensor as T
from pymc3 import DensityDist, Uniform, Normal
from pymc3 import Model
from pymc3 import distributions


def grb_model_sigma(number_events, background_rate, 
              observation_time, volume,  grb_rate,
             efficiency_prior = "uniform"):
    with Model() as model:
        signal_rate = pm.DensityDist('signal_rate', 
                                     logp=lambda value: log_signal_rate(value, number_events, background_rate, observation_time), testval=number_events+2,
                           )

        #volume = pm.Normal("volume", volume, sigma_volume)
        
        n_galaxy = number_mweg(volume)
    
        cbc_rate = pm.Deterministic('cbc_rate', signal_rate / n_galaxy)# * n_galaxy)
        
        grb_rate = (grb_rate /  number_mweg(1e9))
        
        # Allow the efficiency prior to be switched-out
        if efficiency_prior == "uniform":
            efficiency = pm.Uniform('efficiency', 0,1, testval=0.5)
        elif efficiency_prior == "jeffreys":
            efficiency = pm.Beta('efficiency', 0.5, 0.5, testval = 0.3)
        elif isinstance(efficiency_prior, float):
            efficiency = efficiency_prior
        
        def cosangle(cbc_rate, efficiency, grb_rate):
            return T.switch((grb_rate >= cbc_rate*efficiency), -np.Inf, 
                                 (1.0 - ((grb_rate/(cbc_rate*efficiency)))))
        
        costheta = pm.Deterministic('cos_angle', cosangle(cbc_rate, efficiency, grb_rate))

        angle = pm.Deterministic("angle", theano.tensor.arccos(costheta))
        
        return model

horizon_int = 10 #200
events_max = 20

# make a plot of the beaming angle as a function of observation volume against number of detections
# O1 Scenarios
scenarios = []
for events in range(events_max):
    for horizon in np.linspace(hor_start, hor_start+10, horizon_int):

        ##

        volume = np.pi * (4./3.) * (2.26*horizon)**3
        print "Horizon: {}\t Events: {}\t MWEG: {}\t volume: {}".format(horizon, events, number_mweg(volume), volume)

        number_events = events # There were no BNS detections in O1
        background_rate = 0.01 # We take the FAR to be 1/100 yr
        observation_time = 1.0  # Years
        grb_rate = 10.0
        #for prior in priors:
        scenarios.append( grb_model_sigma(number_events, background_rate, observation_time, volume, grb_rate, efficiency_prior=eff_prior))

        ##

traces = []
angles975 = []
angles025 = []
angles500 = []
uppers = []
lowers = []
samples = 500000

priors = ["jeffreys"]

for model in scenarios:
    with model:
        step = pm.Metropolis()
        trace = pm.sample(samples, step)
        traces.append(trace)
        t_data = trace['angle'][trace['angle']>0]
        t_data = t_data[~np.isnan(t_data)]
        t_data = t_data[np.isfinite(t_data)]
        t_data = t_data[50000:]

        
        try:
            lower, upper = pm.stats.hpd(t_data, alpha=0.05, transform=np.rad2deg)
        except ValueError:
            lower, upper = np.nan, np.nan
            print "There weren't enough samples."

        
        angles500.append(np.nanpercentile(trace['angle'][50000:], 50))
        lowers.append(lower)
        uppers.append(upper)
        np.savetxt("{}-upper-{}.dat".format(eff,prior, hor_start), uppers)
        np.savetxt("{}-lower-{}.dat".format(eff_prior, hor_start), lowers)
        np.savetxt("{}-{}.dat".format(eff_prior, hor_start), angles500)

