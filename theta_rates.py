#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@physics.gatech.edu>
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

"""

theta_rates.py

Monte-Carlo demonstration of impact of jet angle on relative rates

Procedure for Nbns events
0) Specify jet distribution
1) Draw a jet angle 
2) Draw an inclination angle
3) If inclination < 0.5 * jet angle, GRB is seen; Ngrb += 1 

"""

from __future__ import division
import os,sys
import numpy as np
import scipy.stats as stats

import matplotlib
from matplotlib import pyplot as pl

def truncparms(low,upp,mu,sigma):
    a = (low - mu) / sigma
    b = (upp - mu) / sigma
    return a, b


def compute_efficiency(k,N,b=True):

    if b:
        # Bayesian treatment
        epsilon=(k+1)/(N+2)
        stdev_epsilon=1.64*np.sqrt(epsilon*(1-epsilon)/(N+3))
    else:
        # Binomial treatment
        if N==0:
            epsilon=0.0
            stdev_epsilon=0.0
        else:
            epsilon=k/N
            stdev_epsilon=1.64*np.sqrt(epsilon*(1-epsilon)/N)
    return (epsilon,stdev_epsilon)


# ------------------------
# Setup
Nbns = 1e5 

# ------------------------
# Distributions

#
# Inclination angle distribution: uniform in cos(iota)
#
iotas = np.arccos( 0 + 1*np.random.rand(Nbns) )
# to degrees
iotas *= 180/np.pi

#
# Jet angle distribution: truncated normal
#
theta_mu=np.arange(5,35,5)
theta_sigma=np.arange(1,16)
theta_low=0.01
theta_upp=90

FracGRB=np.zeros(len(theta_sigma))
deltaFracGRB=np.zeros(len(theta_sigma))

f = pl.subplots()

linestyles=['-', '--', ':']
markers=['s', '^', 'o']

l=0
mk=0
for m,mu, in enumerate(theta_mu):
    print m
    for s,sigma in enumerate(theta_sigma):

        #a, b = truncparms(theta_low, theta_upp, theta_mu, theta_sigma)
        #thetas = stats.truncnorm.rvs(a, b, loc=theta_mu, scale=theta_sigma, size=Nbns)
        a, b = truncparms(theta_low, theta_upp, mu, sigma)
        thetas = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=Nbns)


        # ------------------------
        # Count GRBs
        Ngrbs=sum(iotas <= thetas)
        FracGRB[s], deltaFracGRB[s] = compute_efficiency(Ngrbs,Nbns)

    if l<len(linestyles)-1:
        l+=1
    else:
        mk+=1
        l=0

    # plot
    f[1].errorbar(x=theta_sigma, y=FracGRB*100, yerr=deltaFracGRB*100,
            label=r'$\langle \theta \rangle=%.0f$'%mu, linestyle=linestyles[l],
            marker=markers[mk], color='k', capsize=0.1, markerfacecolor='k')


f[1].set_xlabel(r'$\sigma_{\theta}$ [deg]')
f[1].set_ylabel(r'N$_{\rm{GRB}}$/N$_{\rm{bns}}$ [%]')
#f[1].legend(loc='upper left')
f[1].legend(bbox_to_anchor=(0.8,1.1), ncol=3)
f[1].minorticks_on()

# XXX NOTE: there is a degeneracy in this resulting plot; e.g., pick the 2%
# ratio:  that can correspond to BOTH a N(10,5) AND a N(5,10) distribution!
# Not the end of the world; we still put an upper limit on the mean value -
# i.e., the degenerate value can be LESS than 10 in this case, but NOT more.
#
# So, the plot shows that we are basically insensitive to sigma, but we are
# sensitive to the upper limit on the mean (from looking at pure relative
# numbers of events / rates)


#   print 'Summary:'
#   print 'Nbns=', Nbns
#   print 'theta=N(%.1f,%.1f)'%(theta_mu, theta_sigma)
#   print 'Ngrbs=', Ngrbs
#   print 'FracGRB=%.2f percent'%(100*FracGRB[s])

# -----------------
# Plots
#pl.figure()
#pl.hist(0.5*thetas,100,normed=True,histtype='stepfilled',alpha=0.5)
#pl.hist(iotas,100,normed=True,histtype='stepfilled',alpha=0.5)
pl.show()


