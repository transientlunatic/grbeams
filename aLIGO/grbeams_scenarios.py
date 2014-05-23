#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2013-2014 James Clark <clark@physics.umass.edu>
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
Script to do the ADE rate posterior measurements and hence beaming angle
inferences
"""

from __future__ import division
import os,sys
import numpy as np

import matplotlib
from matplotlib import pyplot as pl

import grbeams_utils

fig_width_pt = 246  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

matplotlib.rcParams.update(
        {'axes.labelsize': 8,
        'text.fontsize':   8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': True,
        'figure.figsize': fig_size,
        'font.family': "serif",
        'font.serif': ["Times"]
        })  

matplotlib.rcParams.update(
        {'savefig1.dpi': 200,
        'xtick.major.size':8,
        'xtick.minor.size':4,
        'ytick.major.size':8,
        'ytick.minor.size':4
        })  
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

__author__ = "James Clark <james.clark@ligo.org>"


#######################
# -*- iLIGO stuff -*- #

#########################
# -*- ADE  -*- #

epochs = [2016, 2022]
predictions = ['re', 'high']

sim_grbs=True
if sim_grbs:
    epsilon=0.5
    theta_jet=10.0

# test case (Rgrb from known efficiency, theta, bns_rate):

# ---------- Priors ------------ #
prior=sys.argv[1]
valid_priors=['delta,0.1', 'delta,0.5', 'delta,1.0', 'uniform']
prior_names={'delta,0.1':'delta_0p1', 'delta,0.5':'delta_0p5',
'delta,1.0':'delta_0p1', 'uniform':'uniform_0p01-1'}
if prior not in valid_priors:
    print >> sys.stderr, "error: invalid prior: ", prior
    sys.exit(-1)

print >> sys.stdout, "\n --- Using %s efficiency prior ---"%prior

linestyles=['-','--',':','.-','-']
markers=['.','^','+','v','*']

for p,prediction in enumerate(predictions):
    print >> sys.stdout, "-------------------"
    print >> sys.stdout, "bns rate: ", prediction

    #f_rate,ax_bns_rate=pl.subplots( figsize=(8,6) )
    #f_angle,ax_jet_angle=pl.subplots( figsize=(8,6) )
    f_rate,ax_bns_rate=pl.subplots( )
    f_angle,ax_jet_angle=pl.subplots( )

    biggest_ul_rate=0.0
    biggest_ul_jet=0.0
    #for p,prediction in enumerate(predictions):
    for e,epoch in enumerate(epochs):
        print >> sys.stdout, "epoch: ", epoch

        # --- Create Observation Scenario
        scenario = grbeams_utils.Scenarios(epoch=epoch, rate_prediction=prediction)
        scenario.compute_posteriors()

        # --- Construct Jet Posteriors
        # Get GRB rate for a test case
        if sim_grbs:
            grb_rate = grbeams_utils.comp_grb_rate(efficiency=epsilon, theta=theta_jet,
                    bns_rate=1e6*scenario.predicted_bns_rate)
            jetpos = grbeams_utils.JetPosterior(scenario,prior,grb_rate=grb_rate)
        else:
            jetpos = grbeams_utils.JetPosterior(scenario,prior)
        #jetpos = grbeams_utils.JetPosterior(scenario,'delta,0.1')
        #jetpos = grbeams_utils.JetPosterior(scenario,'delta,0.5')
        #jetpos = grbeams_utils.JetPosterior(scenario,'delta,1.0')
        #jetpos = grbeams_utils.JetPosterior(scenario,'uniform')

        # --- Plotting
        # Get 90% UL: useful for x-limits in plots

        # *** Rate Posterior ***
        ul_rate = grbeams_utils.alpha_ul(scenario.bns_rate, \
                np.exp(scenario.bns_rate_pdf), alpha=0.99)

        if ul_rate>biggest_ul_rate: biggest_ul_rate = ul_rate

        #label_str='$R_{\mathrm{%s}}$'%prediction
        label_str='Epoch=%s'%str(epoch)

        ax_bns_rate.plot(scenario.bns_rate,np.exp(scenario.bns_rate_pdf), \
                color='k', linestyle=linestyles[e], label=r'%s'%label_str)


        # *** Jet Posterior ***
        ul_jet = grbeams_utils.alpha_ul(jetpos.theta, \
                jetpos.jet_pdf_1D, alpha=0.99)
        if ul_jet>biggest_ul_jet: biggest_ul_jet = ul_jet

        ax_jet_angle.plot(jetpos.theta,jetpos.jet_pdf_1D, \
                color='k', linestyle=linestyles[e], 
                label=r'%s'%label_str)

    print >> sys.stdout, "finalising figures"

    tit_str='Rate: $R_{\mathrm{%s}}$'%prediction
    #ax_bns_rate.set_title('%s'%str(tit_str))
    ax_bns_rate.set_xlabel('BNS Coalescence Rate $R$ [Mpc$^{-3}$ Myr$^{-1}$]')
    ax_bns_rate.set_ylabel('$p(R|N_{\mathrm{det}},T_{\mathrm{obs}},I)$')
    ax_bns_rate.minorticks_on()
    #ax_bns_rate.grid(which='major',color='grey',linestyle='-')
    ax_bns_rate.axvline(scenario.predicted_bns_rate * 1e6, color='r',\
            label="`True' value")
    ax_bns_rate.set_xlim(0,5e6*scenario.predicted_bns_rate)
    ax_bns_rate.legend()
    f_rate.subplots_adjust(bottom=0.2,left=0.15,right=0.925)

    #ax_jet_angle.set_title('%s'%str(tit_str))
    ax_jet_angle.set_xlabel(r'$\theta_{\mathrm{jet}}$')
    ax_jet_angle.set_ylabel(r'$p(\theta_{\mathrm{jet}}|R,I)$')
    ax_jet_angle.minorticks_on()
    #ax_jet_angle.grid(which='major',color='grey',linestyle='-')
    ax_jet_angle.set_xlim(0,60)
    #if prediction=='re': ax_jet_angle.set_xlim(0,60)
    #if prediction=='high': ax_jet_angle.set_xlim(0,20)
    if sim_grbs:
        ax_jet_angle.axvline(theta_jet, color='r', label="`True' value")
    ax_jet_angle.legend()
    f_angle.subplots_adjust(bottom=0.2,top=0.925,left=0.15)

    pl.subplots_adjust(bottom=0.2,top=0.925,left=0.15,right=0.95)

    if sim_grbs:
        f_angle.savefig('angle_%s_%s_sim_theta-%.1f_epsilon-%.1f.eps'%(prediction,prior,theta_jet,epsilon))
    else:
        f_angle.savefig('angle_%s_%s.eps'%(prediction,prior_names[prior]))
    f_rate.savefig('rate_%s.eps'%prediction)

#pl.show()


