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

#from scipy import stats,misc
#import scipy as sci
from scipy.misc import logsumexp

__author__ = "James Clark <james.clark@ligo.org>"

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

class RatePosteriorKnownBG:
    """
    Rate posterior for known background
    """
    def __init__(self,Ntrigs,background_rate,Tobs):

        # observations
        self.Ntrigs=Ntrigs
        self.background_rate=background_rate
        self.Tobs=Tobs

        # GW detection rate posterior
        self.det_rate=np.linspace(sys.float_info.epsilon,2*self.Ntrigs/self.Tobs,1000)
        self.det_rate_pdf=self.comp_pdf_source_rate(self.det_rate)

    def compute_logInvC(self,i):
        """
        Method for Log of Coefficient C^{-1}(i) in gregory poisson rate posterior
        """
        return i*np.log(self.background_rate*self.Tobs) \
                - self.background_rate*self.Tobs - logstirling(i)

    def comp_prob_source_rate(self,source_rate):
        """
        Posterior probability of GW detection rate source_rate
        """
        i = np.arange(self.Ntrigs+1)
        logInvC = logsumexp(self.compute_logInvC(i))
        logC = -1.0*logInvC

        try:
            return logC + self.Ntrigs \
                    * np.log(self.Tobs*self.Tobs*(source_rate+self.background_rate))\
                    - self.Tobs*(source_rate + self.background_rate)\
                    - logstirling(self.Ntrigs)
        except RuntimeWarning:
            return -inf


    def comp_pdf_source_rate(self,source_rate):
        """
        Vectorise the rate pdf calculation in comp_prob_source_rate()
        """

        vprob = np.vectorize(self.comp_prob_source_rate)

        return vprob(source_rate)

class RatePosteriorONOFF:
    """
    Rate posterior for unknown background
    """

    def __init__(self,Non,Noff,Ton,Toff):

        # observations
        self.Non=Non
        self.Noff=Noff
        self.Ton=Ton
        self.Toff=Toff

        # GW detection rate posterior
        self.det_rate=np.linspace(sys.float_info.epsilon,2*self.Non/self.Ton,1000)
        self.det_rate_pdf=self.comp_pdf_source_rate(self.det_rate)

    def compute_logC(self,i):
        """
        Method for Log of Coefficient Ci in gregory poisson rate posterior
        """

        # numerator
        num_time_term = i*np.log(1.0 + self.Ton/self.Toff)
        num_obs_term = logstirling(self.Non+self.Noff-i) \
                - logstirling(self.Non-i)
        log_numerator = num_time_term + num_obs_term

        # denominator
        j=np.arange(self.Non+1)
        den_time_term = j*np.log(1.0 + self.Ton/self.Toff)
        den_obs_term = logstirling(self.Non+self.Noff-j) \
                - logstirling(self.Non-j)
        # denominator is a sum; we do this in log-space
        log_denominator = logsumexp(den_time_term + den_obs_term)

        return log_numerator - log_denominator

    def comp_prob_source_rate(self,source_rate):
        """
        Posterior probability of GW detection rate source_rate
        """
        i=np.arange(self.Non+1)
        log_time_term = np.log(self.Ton) + i*np.log(source_rate*self.Ton) - \
                source_rate*self.Ton - logstirling(i)
        logC = self.compute_logC(i)

        return logsumexp(logC + log_time_term)

    def comp_pdf_source_rate(self,source_rate):
        """
        Vectorise the rate pdf calculation in comp_prob_source_rate()
        """

        vprob = np.vectorize(self.comp_prob_source_rate)

        return vprob(source_rate)


def  main():
    print 'Executing main() of ' + sys.argv[0]

    g = RatePosteriorONOFF(Non=40,Noff=1e-2,Ton=1,Toff=1)
    import pylab as pl
    pl.figure()
    pl.plot(g.det_rate,np.exp(g.det_rate_pdf))

    print np.trapz(np.exp(g.det_rate_pdf),g.det_rate)

    g = RatePosteriorKnownBG(Ntrigs=100,background_rate=1e-2,Tobs=1)
    pl.plot(g.det_rate,np.exp(g.det_rate_pdf))
    pl.show()

if __name__ == "__main__":

    main()
