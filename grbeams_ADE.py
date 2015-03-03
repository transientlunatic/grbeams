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
import cPickle as pickle
import argparse

import matplotlib
from matplotlib import pyplot as pl

import grbeams_utils

from pylal import bayespputils as bppu

if 0:
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

def parse_input():
    """
    Option parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('prior', metavar='pi', type=str, nargs=1, 
            help="type of prior to use for efficiency")

    # --- General
    parser.add_argument('--Rgrb', type=float, default=3e-9, 
            help="Observed rate of sGRBs in local Universe")

    parser.add_argument('--bns-rate', type=str, default='re',
            help="realistic (re) or high (hi) bns rate")

    parser.add_argument('--user-tag', type=str, default='',
            help="string for human-friendly file ID")
    
    # --- For simulating GRBs:
    parser.add_argument('--sim-grbs', action='store_true', default=False, 
        help="synthesise an observed grb rate from specified efficiency, \
                beaming angle and coalesecence rate")
    parser.add_argument('--sim-epsilon', type=float, default=0.5,
            help="GRB efficiency to use for simulated Rgrb")
    parser.add_argument('--sim-theta', type=float, default=10.0,
            help="GRB beaming angle to use for simulated Rgrb")

    args = parser.parse_args()

    # --- Sanity checks
    valid_priors=['delta,0.1', 'delta,0.5', 'delta,1.0', 'uniform', 'jeffreys']
    
    if args.prior[0] not in valid_priors:
        print >> sys.stderr, "error: invalid prior: ", args.prior[0]
        sys.exit(-1)

    #if args.bns_rate[0] not in ['re','hi']:

    
    return args

#########################
# -*- ADE  -*- #

args = parse_input()

epochs = ['2016', '2022']
#epochs = ['2022']
#epochs = ['2016']

# ---------- Priors ------------ #

# this is handy for file naming, so we don't have unwanted characters
prior_names={'delta,0.1':'delta_0p1', 'delta,0.5':'delta_0p5',
'delta,1.0':'delta_1', 'uniform':'uniform_0p01-1','jeffreys':'jeffreys'}

print >> sys.stdout, "\n --- Using %s efficiency prior ---"%args.prior[0]

linestyles=['-','--',':','.-','-']
markers=['.','^','+','v','*']


#for p,prediction in enumerate(predictions):

prediction = args.bns_rate
print >> sys.stdout, "-------------------"
print >> sys.stdout, "bns rate: ", prediction

f_rate,ax_bns_rate=pl.subplots( )
f_angle,ax_jet_angle=pl.subplots( )

#for p,prediction in enumerate(predictions):
for e,epoch in enumerate(epochs):
    print >> sys.stdout, "epoch: ", epoch

    # --- Create Observation Scenario
    scenario = grbeams_utils.Scenarios(epoch=epoch, rate_prediction=prediction)

    # compute rate posteriors:
    scenario.compute_posteriors()

    # --- Construct Jet Posteriors
    # Get GRB rate for a test case
    if args.sim_grbs:
#        if args.Rgrb is not None:
#            grb_rate = args.Rgrb
#        else:
        grb_rate = grbeams_utils.comp_grb_rate(efficiency=args.sim_epsilon,
                theta=args.sim_theta, bns_rate=scenario.predicted_bns_rate)

        thetapos = grbeams_utils.thetaPosterior(scenario, args.prior[0],
                grb_rate=grb_rate, sim_theta=args.sim_theta)
    else:
        thetapos = grbeams_utils.thetaPosterior(scenario, args.prior[0],
                grb_rate=args.Rgrb)


    # Get posterior samples and kde
    thetapos.sample_theta_posterior()

    # select theta posterior bandwidth for KDE
    theta_bw = 1.06*np.std(thetapos.theta_samples)*\
            len(thetapos.theta_samples)**(-1./5)

    # --- Plotting
    # Get 90% UL: useful for x-limits in plots
    theta_bin_size = 3.5*np.std(thetapos.theta_samples) \
            / len(thetapos.theta_samples)**(1./3)
    theta_bins = np.arange(thetapos.theta_range.min(), thetapos.theta_range.max(),
            theta_bin_size)

    thetapos.get_theta_pdf_kde(bandwidth=theta_bw)

    # *** Rate Posterior ***
    label_str='Epoch=%s'%str(epoch)
    ax_bns_rate.plot(scenario.bns_rate,np.exp(scenario.bns_rate_pdf), \
            color='k', linestyle=linestyles[e], label=r'%s'%label_str)

    # *** Jet Posterior ***
    ax_jet_angle.hist(thetapos.theta_samples, bins=theta_bins, normed=True,
            histtype='stepfilled', alpha=0.5)
    ax_jet_angle.plot(thetapos.theta_grid,thetapos.theta_pdf_kde, \
            color='k', linestyle=linestyles[e], 
            label=r'%s'%label_str)

    # --- bppu estimates:
    ax_jet_angle.axvline(thetapos.theta_median, color='k',
            linestyle=linestyles[e])
    ax_jet_angle.axvline(thetapos.theta_bounds[0], color='k',
            linestyle=linestyles[e])
    ax_jet_angle.axvline(thetapos.theta_bounds[1], color='k',
            linestyle=linestyles[e])

    if args.sim_grbs:
        # just dump the posterior object - that has everything we need
        f_angle_pickle = file('angle_%s_%s_%s_sim_theta-%.1f_epsilon-%.1f%s.pickle'%(\
                prediction,epoch,args.prior[0],args.sim_theta,args.sim_epsilon,args.user_tag),'wb')
        pickle.dump(thetapos.theta_pos,f_angle_pickle)
    else:
        f_angle_pickle = file('angle_%s_%s_%s%s.pickle'%(\
                prediction,epoch,prior_names[args.prior[0]],args.user_tag),'wb')
        # just dump the posterior object - that has everything we need
        pickle.dump(thetapos.theta_pos,f_angle_pickle)

    f_angle_pickle.close()



print >> sys.stdout, "finalising figures"

tit_str='Rate: $R_{\mathrm{%s}}$'%prediction
ax_bns_rate.set_xlabel('BNS Coalescence Rate $R$ [Mpc$^{-3}$ Myr$^{-1}$]')
ax_bns_rate.set_ylabel('$p(R|N_{\mathrm{det}},T_{\mathrm{obs}},I)$')
ax_bns_rate.minorticks_on()
ax_bns_rate.axvline(scenario.predicted_bns_rate, color='k',\
        label="`True' value")
ax_bns_rate.set_xlim(0,5*scenario.predicted_bns_rate) # multiply by 5 to get
                                                      # well beyond the 'true' values
ax_bns_rate.legend()
f_rate.subplots_adjust(bottom=0.15,left=0.1,right=0.925)
f_rate.tight_layout()

ax_jet_angle.set_xlabel(r'$\theta_{\mathrm{jet}}$')
ax_jet_angle.set_ylabel(r'$p(\theta_{\mathrm{jet}}|R,I)$')
ax_jet_angle.minorticks_on()
#ax_jet_angle.set_xlim(0,60)

if args.sim_grbs:
    ax_jet_angle.axvline(args.sim_theta, color='g', label="`True' value")

ax_jet_angle.legend()
f_angle.subplots_adjust(bottom=0.1,top=0.925,left=0.1,right=0.925)
f_angle.tight_layout()

#pl.subplots_adjust(bottom=0.2,top=0.925,left=0.15,right=0.95)

if args.sim_grbs:
    f_angle.savefig('angle_%s_%s_sim_theta-%.1f_epsilon-%.1f%s.pdf'%(\
            prediction,args.prior[0],args.sim_theta,args.sim_epsilon,args.user_tag))
    f_angle.savefig('angle_%s_%s_sim_theta-%.1f_epsilon-%.1f%s.eps'%(\
            prediction,args.prior[0],args.sim_theta,args.sim_epsilon,args.user_tag))
    f_angle.savefig('angle_%s_%s_sim_theta-%.1f_epsilon-%.1f%s.png'%(\
            prediction,args.prior[0],args.sim_theta,args.sim_epsilon,args.user_tag))



else:
    f_angle.savefig('angle_%s_%s%s.pdf'%(\
            prediction,prior_names[args.prior[0]],args.user_tag))
    f_angle.savefig('angle_%s_%s%s.eps'%(\
            prediction,prior_names[args.prior[0]],args.user_tag))
    f_angle.savefig('angle_%s_%s%s.png'%(\
            prediction,prior_names[args.prior[0]],args.user_tag))

f_rate.savefig('rate_%s%s.eps'%(prediction,args.user_tag))
f_rate.savefig('rate_%s%s.pdf'%(prediction,args.user_tag))
f_rate.savefig('rate_%s%s.png'%(prediction,args.user_tag))
f_rate_pickle = file('rate_%s%s.pickle'%(prediction,args.user_tag),'wb')
pickle.dump((scenario.bns_rate, scenario.bns_rate_pdf),f_rate_pickle)
f_rate_pickle.close()

#pl.close(2)
#pl.show()


