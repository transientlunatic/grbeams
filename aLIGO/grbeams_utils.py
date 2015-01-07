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
import os,sys,time
import numpy as np
import emcee

from scipy.misc import logsumexp#,factorial
from scipy import stats, optimize
from sklearn.neighbors.kde import KernelDensity

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

def comp_grb_rate(efficiency, theta, bns_rate):
    """
    Computes the GRB rate:
    Rgrb = epsilon*(1-cos(theta))*Rbns
    """
    return efficiency*(1-np.cos(theta/180.0 * np.pi))*bns_rate

def cbc_rate_from_theta(grb_rate,theta,efficiency):
    """
    Returns Rcbc = Rgrb / (1-cos(theta))
    """
    return grb_rate / ( efficiency*(1.-np.cos(theta * np.pi / 180)) )

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)
    

def characterise_dist(x,y,alpha):
    """
    Return the alpha-confidence limits about the median, and the median of the
    PDF whose y-values are contained in y and x-values in x
    """

    # Peak
    posmax = x[np.argmax(y)]

    # CDF:
    cdf = np.cumsum(y)
    cdf /= max(cdf)
    
    # Fine-grained values (100x finer than the input):
    x_interp = np.arange(min(x), max(x), np.diff(x)[0]/100.0)
    cdf_interp = np.interp(x_interp, x, cdf)

    # median
    median_val = np.interp(0.5,cdf_interp,x_interp)

    # alpha-ocnfidence width about the median
    q1 = (1-alpha)/2.0
    q2 = (1+alpha)/2.0
    low_bound = np.interp(q1,cdf_interp,x_interp)
    upp_bound = np.interp(q2,cdf_interp,x_interp)

    # alpha-confidence *upper* limit
    low_limit = np.interp(alpha,(1-cdf_interp)[::-1],x_interp[::-1])
    upp_limit = np.interp(alpha,cdf_interp,x_interp)

    #return [low_bound,upp_bound],median_val,[low_limit,upp_limit]
    return [low_limit,upp_limit], median_val, posmax

def compute_confints(x_axis,pdf,alpha):
    """
    Confidence Intervals From KDE
    """

    # Make sure PDF is correctly normalised
    pdf /= np.trapz(pdf,x_axis)

    # --- initialisation
    peak = x_axis[np.argmax(pdf)]

    # Initialisation
    area=0.

    i=0 
    j=0 

    x_axis_left=x_axis[(x_axis<peak)][::-1]
    x_axis_right=x_axis[x_axis>peak]

    while area <= alpha:

        x_axis_current=x_axis[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]
        pdf_current=pdf[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]

        area=np.trapz(pdf_current,x_axis_current)

        if i<len(x_axis_left)-1: i+=1
        if j<len(x_axis_right)-1: j+=1

    low_edge, upp_edge = x_axis_left[i], x_axis_right[j]

    return ([low_edge,upp_edge],peak,area)


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
        valid_epochs=['2015', '2016', '2017', '2019', '2022']
        valid_rate_predictions=['low', 're', 'high']
        if epoch not in valid_epochs:
            print >> sys.stderr, "Error, epoch must be in ", valid_epochs
            sys.exit()
        if rate_prediction not in valid_rate_predictions:
            print >> sys.stderr, "Error, rate_prediction must be in ",\
                    valid_rate_predictions
            sys.exit()

        # --- get characteristics of this scenario
        self.duty_cycle=self.get_duty_cycle(epoch)
        self.predicted_bns_rate=self.get_predicted_bns_rate(rate_prediction)
        self.Tobs=self.get_run_length(epoch)
        self.far=self.get_far()

        # --- compute derived quantities for this scenario
        self.bns_search_volume=self.get_bns_search_volume(epoch)
        self.Ngws=self.compute_num_detections()

    def get_duty_cycle(self,epoch):
        """
        return the network duty cycle for this observing scenario
        """
        if epoch=='2022':
            return 1.0
        else:
            return 0.5

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
        self.bns_rate=self.det_rate / (self.bns_search_volume / self.Tobs)
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
        det_rate = bns_rate * (self.bns_search_volume / self.Tobs)

        # Set rate pdf=0 unless rate>=0
        bns_rate_pdf = np.zeros(np.shape(det_rate)) - np.inf
        non_zero_idx = np.argwhere(det_rate>=0.0)
        bns_rate_pdf[non_zero_idx] = \
                self.det_rate_posterior.source_rate_pdf(det_rate) \
                + np.log(self.bns_search_volume / self.Tobs)

        return bns_rate_pdf 

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
        # Divide by 1e6 for Myr -> yr
        rates={'low':0.01/1e6, 're':1.0/1e6, 'high':10/1e6}

        return rates[prediction]

    def get_bns_range(self,epoch):
        """
        The BNS range distances taken from the ADE scenarios document.
        Includes the range in values.  We only handle aLIGO here but it
        shouldn't be difficult to change this.
        """
        bns_ranges={'2015':np.array([40,80]), '2016':np.array([80,120]),
                '2017':np.array([120,170]), '2019':np.array([200]),
                '2022':np.array([200])}

        return bns_ranges[epoch]

    def get_run_length(self,epoch):
        """
        The projected science run durations in years
        """
        run_lengths={'2015':1./12, '2016':6./12, '2017':9./12, '2019':1., '2022':1.}

        return run_lengths[epoch]

    def get_bns_search_volume(self,epoch):
        """
        The BNS search volume (Mpc^3 yr) at rho_c=12 from the ADE scenarios
        document.  We could quite easily compute ranges and hence
        volumes ourselves if we wanted but note that these account for expected
        duty cycles.
        """
    #    volumes={2015:np.mean([0.4e5,3e5]), 2016:np.mean([0.6e6,2e6]),
    #            2017:np.mean([3e6,10e6]), 2019:np.mean([2e7]),
    #            2022:np.mean([4e7])}
    #    return volumes[epoch]
        bns_range = self.get_bns_range(epoch)
        if len(bns_range)>1: bns_range = np.mean(bns_range)
        return self.duty_cycle * self.Tobs * 4.0 * np.pi * (bns_range**3) / 3

class thetaPosterior:

    def __init__(self, observing_scenario, efficiency_prior='delta,1.0',
            grb_rate=10/1e9):

        # Sanity check efficiency prior
        valid_priors = ['delta,0.01','delta,0.1', 'delta,0.5', 'delta,1.0',
                'uniform', 'jeffreys']

        if efficiency_prior not in valid_priors:
            print >> sys.stderr, "ERROR, %s not recognised"%efficiency_prior
            print >> sys.stderr, "valid priors are: ", valid_priors
            sys.exit()
        else:
            self.efficiency_prior = efficiency_prior

        # --- MCMC Setup
        # Dimensionality to increment with every non-delta function prior which
        # gets added
        self.ndim = 1

        # --- Prior Setup
        self.theta_range = np.array([0.0,90.0])
        if efficiency_prior in ['delta,0.01','delta,0.1','delta,0.5','delta,1.0']:
            self.efficiency_range = float(efficiency_prior.split(',')[1])
        #elif efficiency_prior == 'uniform':
        elif efficiency_prior in ['uniform', 'jeffreys']:
            # increase dimensionality of parameter space and set efficiency
            # prior range
            self.ndim += 1
            self.efficiency_range = np.array([0.0,1.0])
#        elif efficiency_prior == 'jeffreys':
#            self.ndim += 1
#            self.efficiency_range = np.array([0.001,0.999])

        # --- Astro configuration
        # grb_rate is in units of Mpc^-3 yr^-1 but the input is Gpc-3 yr-1
        self.grb_rate = grb_rate #* 1e-9 
        self.scenario = observing_scenario

    def get_theta_pdf_kde(self):

        self.theta_grid  = np.arange(self.theta_range[0], self.theta_range[1], 0.01)
        self.theta_pdf_kde  = kde_sklearn(x=self.theta_samples,
                x_grid=self.theta_grid, bandwidth=1.5, algorithm='kd_tree') 
        self.theta_bounds, self.theta_median, self.theta_posmax = \
                characterise_dist(self.theta_grid, self.theta_pdf_kde, 0.9)

    def sample_theta_posterior(self, nburnin=100, nsamp=500, nwalkers=100):
        """
        Use emcee ensemble sampler to draw samples from the ndim parameter space
        comprised of (theta, efficiency, delta_theta, ...) etc

        The dimensionality is determined in __init__ based on the priors used

        The probability function used is the comp_theta_prob() method
        """

        if 'delta' in self.efficiency_prior:

            # Starting points for walkers
            p0 = (max(self.theta_range)-min(self.theta_range)) *\
                    np.random.rand(self.ndim * nwalkers).reshape((nwalkers, self.ndim))

            # Inititalize sampler
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim,
                    self.comp_theta_prob, args=[float(self.efficiency_prior.split(',')[1])])

        else:

            # Starting points for walkers

            theta0 = (max(self.theta_range)-min(self.theta_range)) *\
                    np.random.rand(nwalkers)
            
            efficiency0 = (max(self.efficiency_range)-min(self.efficiency_range)) *\
                    np.random.rand(nwalkers)

            p0 = np.transpose(np.array([theta0, efficiency0]))

            # Inititalize sampler

            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim,
                    self.comp_theta_prob_nparam)

        # Burn-in
        pos, prob, state = self.sampler.run_mcmc(p0, nburnin)
        self.sampler.reset()

        # Draw samples
        self.sampler.run_mcmc(pos, nsamp)

        # 1D array with samples for convenience
        if self.ndim==1:
            self.theta_samples = np.concatenate(self.sampler.flatchain)
        else:
            self.theta_samples = self.sampler.flatchain[:,0]
            self.efficiency_samples = self.sampler.flatchain[:,1]

    def comp_theta_prob_nparam(self, x, fixed_args=None):
        return self.comp_theta_prob(theta=x[0], efficiency=x[1])

            
    def comp_theta_prob(self,theta,efficiency):
        """
        Perform the rate->theta posterior transformation.

        Here's the procedure:
        1) Given an efficiency and theta angle, find the corresponding cbc rate
        according to Rcbc = Rgrb / (1-cos(theta))
        2) evaluate rate posterior at this value of the cbc rate
        3) The theta angle posterior is then just jacobian * rate
        posterior[rate=rate(theta)]
        """
        #print efficiency, self.comp_efficiency_prob(efficiency)
        if (theta>=min(self.theta_range)) and \
                (theta<max(self.theta_range)):

            # Get BNS rate from theta, efficiency
            bns_rate = cbc_rate_from_theta(self.grb_rate, theta, efficiency)

            # Get value of rate posterior at this rate
            bns_rate_pdf = self.scenario.comp_bns_rate_pdf(bns_rate)

            # Compute jacobian
            jacobian = self.compute_jacobian(efficiency,theta)

            theta_prob = bns_rate_pdf + np.log(jacobian) \
                    + np.log(self.comp_efficiency_prob(efficiency))


        else:
            # outside of prior ranges
            theta_prob = -np.inf

        return theta_prob

    def compute_jacobian(self,efficiency,theta):
        """
        Compute the Jacboian for the transformation from rate to angle
        """

        denom=efficiency*(np.cos(theta * np.pi/180)-1)
        return abs(2.0*self.grb_rate * np.sin(theta * np.pi / 180.0) /
                (denom*denom) )


    def comp_efficiency_prob(self,efficiency):
        """
        Prior on the BNS->GRB efficiency
        """

        prior_type = self.efficiency_prior.split(',')[0]

        if prior_type == 'delta':
            # delta function prior centered at efficiency=prior_params

            if efficiency == float(self.efficiency_prior.split(',')[1]):
                return 1.0
            else:
                return 0.0

        elif prior_type == 'uniform':
            # linear uniform prior

            if (efficiency>=min(self.efficiency_range)) and (efficiency<max(self.efficiency_range)):
                return 1./(max(self.efficiency_range)-min(self.efficiency_range))
            else:
                return 0.0

        elif prior_type == 'jeffreys':
            prior_dist = stats.beta(0.5,0.5)
            if (efficiency>=min(self.efficiency_range)) and (efficiency<max(self.efficiency_range)):
                return prior_dist.pdf(efficiency)
            else:
                return 0.0


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


