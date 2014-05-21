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

from matplotlib import pyplot as pl

import grbeams_utils

__author__ = "James Clark <james.clark@ligo.org>"


#######################
# -*- iLIGO stuff -*- #

#########################
# -*- ADE  -*- #

#epochs = [2015, 2016, 2017, 2019, 2022]
#epochs = [2016, 2022]
epochs = [2016]
#predictions = ['low', 're', 'high']
#predictions = ['re', 'high']
predictions = ['high']
linestyles=['-','--',':','.-','-']
markers=['.','^','+','v','*']

for p,prediction in enumerate(predictions):

    f_rate,ax_bns_rate=pl.subplots( figsize=(8,6) )
    f_angle,ax_jet_angle=pl.subplots( figsize=(8,6) )

    biggest_ul_rate=0.0
    biggest_ul_jet=0.0
    #for p,prediction in enumerate(predictions):
    for e,epoch in enumerate(epochs):

        # --- Create Observation Scenario
        scenario = grbeams_utils.Scenarios(epoch=epoch, rate_prediction=prediction)
        scenario.compute_posteriors()

        # Only continue evaluation if Ndetections>=1
        #if scenario.Ngws < 1: continue

        # --- Construct Jet Posteriors
        #jetpos = grbeams_utils.JetPosterior(scenario,'delta,1.0')
        jetpos = grbeams_utils.JetPosterior(scenario,'uniform')

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
        ax_bns_rate.axvline(scenario.predicted_bns_rate * 1e6, color='r',\
                label="`True' value")

        # *** Jet Posterior ***
        ul_jet = grbeams_utils.alpha_ul(jetpos.theta, \
                jetpos.jet_pdf_1D, alpha=0.99)
        if ul_jet>biggest_ul_jet: biggest_ul_jet = ul_jet

        ax_jet_angle.plot(jetpos.theta,jetpos.jet_pdf_1D, \
                color='k', linestyle=linestyles[e], 
                label=r'%s'%label_str)

    tit_str='Rate: $R_{\mathrm{%s}}$'%prediction
    ax_bns_rate.set_title('%s'%str(tit_str))
    ax_bns_rate.legend()
    ax_bns_rate.set_xlabel('BNS Coalescence Rate $R$ [Mpc$^{-3}$ Myr$^{-1}$]')
    ax_bns_rate.set_ylabel('$p(R|N_{\mathrm{det}},T_{\mathrm{obs}},I)$')
    ax_bns_rate.minorticks_on()
    ax_bns_rate.grid(which='major',color='grey',linestyle='-')

    ax_jet_angle.set_title('%s'%str(tit_str))
    ax_jet_angle.legend()
    ax_jet_angle.set_xlabel(r'$\theta_{\mathrm{jet}}$')
    ax_jet_angle.set_ylabel(r'$p(\theta_{\mathrm{jet}}|R,I)$')
    ax_jet_angle.minorticks_on()
    ax_jet_angle.grid(which='major',color='grey',linestyle='-')

    f_rate.savefig('rate_%s-%s.eps'%(prediction,epoch))
    f_angle.savefig('angle_%s-%s.eps'%(prediction,epoch))

pl.subplots_adjust(bottom=0.1,top=0.925)
#pl.show()


