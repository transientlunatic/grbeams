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
import scipy as sci

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

class RatePosterior:
    """
    """

    def __init__(self,Non,Noff,Ton,Toff):

        # observations
        self.Non=Non
        self.Noff=Noff
        self.Ton=Ton
        self.Toff=Toff

        # GW detection rate posterior
        self.source_rate=np.linspace(0,2*self.Non/self.Ton,1000)
        self.source_rate_pdf=self.comp_pdf_source_rate(self.source_rate)

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
        j=np.arange(self.Non)
        den_time_term = j*np.log(1.0 + self.Ton/self.Toff)
        den_obs_term = logstirling(self.Non+self.Noff-j) \
                - logstirling(self.Non-j)
        # denominator is a sum; we do this in log-space
        log_denominator = sci.misc.logsumexp(den_time_term + den_obs_term)

        # full expression
        logC = log_numerator - log_denominator

        return logC

    def comp_prob_source_rate(self,source_rate):
        """
        Posterior probability of GW detection rate source_rate
        """
        i=np.arange(self.Non)
        log_time_term = np.log(self.Ton) + i*np.log(source_rate*self.Ton) - \
                source_rate*self.Ton - logstirling(i)
        logC = self.compute_logC(i)

        # again, we can do the summation in log-space
        sci.misc.logsumexp(logC + log_time_term)

        return sci.misc.logsumexp(logC + log_time_term)

    def comp_pdf_source_rate(self,source_rate):
        """
        Compute the posterior probability density function on the rate of GW
        detections as a function of rate.  Takes the number of on-source events
        as upper bound on rate.
        """

        # vectorise the rate calculation
        vprob = np.vectorize(self.comp_prob_source_rate)

        return vprob(source_rate)


def  main():
    print 'Executing main() of ' + sys.argv[0]

    g = RatePosterior(Non=24,Noff=9,Ton=3,Toff=3)
    import pylab as pl
    pl.figure()
    pl.plot(g.source_rate,np.exp(g.source_rate_pdf))
    pl.show()

if __name__ == "__main__":

    main()
