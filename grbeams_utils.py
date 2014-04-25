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
"""

from __future__ import division
import os,sys
import numpy as np

from scipy.misc import logsumexp#,factorial
from scipy import stats, optimize

__author__ = "James Clark <james.clark@ligo.org>"

epsilon = sys.float_info.epsilon
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ---- USEFUL FUNCTIONS ---- #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def logstirling(n):
    """
    Stirling's approximation for log(n!)
    """
    try: 
        # Vectorise where possible
        result=np.zeros(len(n))
        idx=np.argwhere(abs(n)>0)
        result[idx]=n[idx]*np.log(n[idx]) - n[idx] + 0.5*np.log(2*np.pi*n[idx])
    except TypeError:
        # TypeError in len(n) indicates a scalar
        if n==0: result=0
        else: result = n*np.log(n) - n + 0.5*np.log(2*np.pi*n)

    return result

def alpha_ul(x,y,alpha=0.9):
    """
    Compute the alpha (e.g., 90%) upper limit from the posterior pdf contained
    in y for the values in x
    """
    alphas=[]
    for xval in x:
        alphas.append(np.trapz(y[x<xval],x[x<xval]))
    return np.interp(alpha, alphas, x)

#def rate_fit(x,a,s,l):
#    return stats.gamma.pdf(x,a,loc=l,scale=s)

from scipy.misc import factorial
def rate_fit(x,C,T,b,n):
    return C*T*( (x+b)*T )**n * np.exp(-(x+b)*T) / factorial(n)

def fit_rate(x,y):
    popt,_ = optimize.curve_fit(rate_fit, x, y)
    return popt
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ---- CLASS DEFS ---- #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class RatePosteriorKnownBG:
    """
    Rate posterior for known background
    """
    def __init__(self,Ntrigs,far,Tobs):

        # Observations
        self.Ntrigs=Ntrigs
        self.far=far
        self.Tobs=Tobs

    def compute_logInvC(self,i):
        """
        Method for Log of Coefficient C^{-1}(i) in gregory poisson rate posterior
        """
        return i*np.log(self.far*self.Tobs) \
                - self.far*self.Tobs - logstirling(i)

    def source_rate_prob(self,source_rate):
        """
        Posterior probability of GW detection rate source_rate
        """
        i = np.arange(self.Ntrigs+1)
        logInvC = logsumexp(self.compute_logInvC(i))
        logC = -1.0*logInvC

        try:
            return logC + np.log(self.Tobs) \
                    + self.Ntrigs * np.log(self.Tobs*(source_rate+self.far))\
                    - self.Tobs*(source_rate + self.far)\
                    - logstirling(self.Ntrigs)
        except RuntimeWarning:
            return -inf


    def source_rate_pdf(self,source_rate):
        """
        Vectorise the rate pdf calculation in comp_prob_source_rate()
        """
 
        vprob = np.vectorize(self.source_rate_prob)
 
        return vprob(source_rate)


class Scenarios:
    """
    Class with the observing scenario information
    """

    def __init__(self,epoch,rate_prediction):

        # --- sanity checks
        valid_epochs=[2015, 2016, 2017, 2019, 2022]
        valid_rate_predictions=['low', 're', 'high']
        if epoch not in valid_epochs:
            print >> sys.stderr, "Error, epoch must be in ", valid_epochs
            sys.exit()
        if rate_prediction not in valid_rate_predictions:
            print >> sys.stderr, "Error, rate_prediction must be in ",\
                    valid_rate_predictions
            sys.exit()

        # --- get characteristics
        self.predicted_bns_rate=self.get_predicted_bns_rate(rate_prediction)
        self.bns_search_volume=self.get_bns_search_volume(epoch)
        self.Tobs=self.get_run_length(epoch)
        self.Ngws=self.compute_num_detections()
        self.far=self.get_far()

    def compute_posteriors(self):
        """
        Compute the rate posteriors. Get the coalescence rate from the detection
        rate and the search volume.

        First, create an instance of the RatePosterior class, then use this to
        evalatue the pdfs for the detection and coalescence rate posteriors at
        specified values.  This lets us produce pre-generated plot data, while
        retaining the flexibility to evaluate the posteriors at any given value.
        """

        # --- detection rate posterior instance
        self.det_rate_posterior = RatePosteriorKnownBG(Ntrigs=self.Ngws, far=self.far,
                Tobs=self.Tobs)

        # detection rate posterior arrays
        self.det_rate = np.linspace(sys.float_info.epsilon,10*self.Ngws/self.Tobs,1000)
        self.det_rate_pdf = self.comp_det_rate_pdf(self.det_rate)

        # BNS coalescence rate posterior arrays for rate in / Mpc^3 / Myr.
        self.bns_rate=1e6*self.det_rate / (self.bns_search_volume / self.Tobs)
        self.bns_rate_pdf = self.comp_bns_rate_pdf(self.bns_rate)

    def comp_det_rate_pdf(self, det_rate):
        """
        Evaluate the detection rate posterior pdf at det_rate
        """
        return self.det_rate_posterior.source_rate_pdf(self.det_rate)

    def comp_bns_rate_pdf(self, bns_rate):
        """
        Evaluate the bns rate posterior pdf at bns_rate: 
        p(bns_rate) = p(det_rate) |d det_rate / d bns_rate|
        """ 
        # Take care since search volume is normalised to actual run time
        det_rate = bns_rate/1e6 * (self.bns_search_volume / self.Tobs)
        return self.det_rate_posterior.source_rate_pdf(det_rate) \
                + np.log(self.bns_search_volume / self.Tobs) - np.log(1e6)

    def get_far(self):
        """
        The expected background rate in years^{-1}.  Hardcode this for now but
        note that it wouldn't be too hard to change just by loading in the data
        from the cbc far figure (fig 3) in the ADE scenarios document
        """
        return 1e-2

    def compute_num_detections(self):
        return self.predicted_bns_rate * self.bns_search_volume #* self.Tobs

    def get_predicted_bns_rate(self,prediction):
        """
        Dictionary of bns coalescence rates per Mpc^3 per yr
        """
        rates={'low':0.01/1e6, 're':1.0/1e6, 'high':10/1e6}

        return rates[prediction]

    def get_bns_search_volume(self,epoch):
        """
        The BNS search volume (Mpc^3 yr) at rho_c=12 from the ADE scenarios
        document.  We could quite easily compute horizon distances and hence
        volumes ourselves if we wanted but note that these account for expected
        duty cycles.
        """
        volumes={2015:np.mean([0.4e5,3e5]), 2016:np.mean([0.6e6,2e6]),
                2017:np.mean([3e6,10e6]), 2019:np.mean([2e7]),
                2022:np.mean([4e7])}
        return volumes[epoch]

    def get_run_length(self,epoch):
        """
        The projected science run durations in years
        """
        run_lengths={2015:1./12, 2016:6./12, 2017:9./12, 2019:1., 2022:1.}

        return run_lengths[epoch]

class JetPosterior:

    def __init__(self, observing_scenario, efficiency_prior='delta,1.0',
            grb_rate=10e6/1e9):

        # Input
        #self.rate_posterior = rate_posterior
        self.efficiency_prior = efficiency_prior
        self.grb_rate = grb_rate 
        self.scenario = observing_scenario

        # Generate efficiency prior (this part for plotting)
        if efficiency_prior in ['delta,0.1','delta,0.5','delta,1.0']:
            self.efficiency = np.array([float(efficiency_prior.split(',')[1])])
        else:
            self.efficiency = np.linspace(0.01,1,5)
        self.efficiency_pdf = self.comp_efficiency_pdf(self.efficiency)

        # Compute jet posterior
        self.theta = np.linspace(0.01,90,5000)
        self.jeteff_pdf_2D = self.comp_jeteff_pdf_2D(self.theta,self.efficiency)
        self.jet_pdf_1D = self.comp_jet_pdf_1D(self.jeteff_pdf_2D)

    def comp_efficiency_pdf(self,efficiency):
        """
        Vectorized version of comp_efficiency_prob()
        """
        vfunc = np.vectorize(self.comp_efficiency_prob)
        return vfunc(efficiency)

    def comp_efficiency_prob(self,efficiency):
        """
        Prior on the BNS->GRB efficiency
        """
        valid_priors = ['delta,0.1','delta,0.5','delta,1.0']
        if self.efficiency_prior not in valid_priors:
            print >> sys.stderr, "ERROR, %s not recognised"%self.efficiency_prior
            print >> sys.stderr, "valid priors are: ", valid_priors
            sys.exit()

        prior_type = self.efficiency_prior.split(',')[0]
        prior_params = self.efficiency_prior.split(',')[1]

        if prior_type == 'delta':
            if efficiency == float(prior_params):
                return 1.0
            else:
                return 0.0

    def comp_jet_pdf_1D(self,jeteff_pdf_2D):
        """
        Compute the 1D marginal distribution on the jet angle
        """
        marginal_jet_pdf = np.sum(np.exp(jeteff_pdf_2D),axis=1)
        return marginal_jet_pdf / np.trapz(marginal_jet_pdf, self.theta)

    def comp_jeteff_pdf_2D(self,theta,efficiency):
        """
        Vectorize the jet posterior evaluation
        """
        # XXX OPTIMISE THIS XXX
        #Theta, Efficiency = np.meshgrid(theta,efficiency)
        #jet_pdf = self.comp_jet_prob(Theta,Efficiency)

        jet_pdf = np.empty(shape=(len(theta),len(efficiency)))
        for e in xrange(len(efficiency)):
            print e
            jet_pdf[:,e] = self.comp_jet_prob(theta,efficiency[e])

        return jet_pdf

    def comp_jet_prob(self,theta,efficiency):
        """
        Perform the rate->jet posterior transformation.

        Here's the procedure:
        1) Given an efficiency and jet angle, find the corresponding cbc rate
        according to Rcbc = Rgrb / (1-cos(theta))
        2) evaluate rate posterior at this value of the cbc rate
        3) The jet angle posterior is then just jacobian * rate
        posterior[rate=rate(theta)]
        """

        # Get BNS rate from theta, efficiency
        bns_rate = self.rateFromTheta(theta,efficiency)

        # Get value of rate posterior at this rate
        bns_rate_pdf = self.scenario.comp_bns_rate_pdf(bns_rate)

        # Compute jacobian
        jacobian = self.compute_jacobian(efficiency,theta)

        return bns_rate_pdf + np.log(jacobian) \
                + np.log(self.comp_efficiency_prob(efficiency))

    def compute_jacobian(self,efficiency,theta):
        """
        Compute the Jacboian for the transformation from rate to angle
        """
        denom=efficiency*(np.cos(theta * np.pi/180)-1)
        return abs(2*self.grb_rate * np.sin(theta * np.pi / 180) / denom*denom)

    def rateFromTheta(self,theta,efficiency):
        """
        Returns Rcbc = Rgrb / (1-cos(theta))
        """
        return self.grb_rate / ( efficiency*(1.-np.cos(theta * np.pi / 180)) )


def  main():

    print 'Executing main() of ' + sys.argv[0]

    import pylab as pl

    NgwDet=float(sys.argv[1])
    Tobs=float(sys.argv[2])

    g = RatePosteriorKnownBG(Ntrigs=NgwDet,far=1e-2,Tobs=Tobs)
    pl.figure()
    pl.plot(g.det_rate,np.exp(g.det_rate_pdf))
    pl.show()

    #print np.trapz(np.exp(g.det_rate_pdf),g.det_rate)

if __name__ == "__main__":

    main()

#   class RatePosteriorONOFF:
#       """
#       Rate posterior for unknown background
#       """
#
#       def __init__(self,Non,Noff,Ton,Toff):
#
#           # observations
#           self.Non=Non
#           self.Noff=Noff
#           self.Ton=Ton
#           self.Toff=Toff
#
#           # GW detection rate posterior
#           self.det_rate=np.linspace(sys.float_info.epsilon,10*self.Non/self.Ton,1000)
#           self.det_rate_pdf=self.comp_pdf_source_rate(self.det_rate)
#
#       def compute_logC(self,i):
#           """
#           Method for Log of Coefficient Ci in gregory poisson rate posterior
#           """
#
#           # numerator
#           num_time_term = i*np.log(1.0 + self.Ton/self.Toff)
#           num_obs_term = logstirling(self.Non+self.Noff-i) \
#                   - logstirling(self.Non-i)
#           log_numerator = num_time_term + num_obs_term
#
#           # denominator
#           j=np.arange(self.Non+1)
#           den_time_term = j*np.log(1.0 + self.Ton/self.Toff)
#           den_obs_term = logstirling(self.Non+self.Noff-j) \
#                   - logstirling(self.Non-j)
#           # denominator is a sum; we do this in log-space
#           log_denominator = logsumexp(den_time_term + den_obs_term)
#
#           return log_numerator - log_denominator
#
#       def comp_prob_source_rate(self,source_rate):
#           """
#           Posterior probability of GW detection rate source_rate
#           """
#           i=np.arange(self.Non+1)
#           log_time_term = np.log(self.Ton) + i*np.log(source_rate*self.Ton) - \
#                   source_rate*self.Ton - logstirling(i)
#           logC = self.compute_logC(i)
#
#           return logsumexp(logC + log_time_term)
#
#       def comp_pdf_source_rate(self,source_rate):
#           """
#           Vectorise the rate pdf calculation in comp_prob_source_rate()
#           """
#
#           vprob = np.vectorize(self.comp_prob_source_rate)
#
#       return vprob(source_rate)


