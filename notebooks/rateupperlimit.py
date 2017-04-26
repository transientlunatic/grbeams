from __future__ import print_function

import numpy as np
import copy
import glob
import h5py

import scipy.special as special
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import pycbc.sensitivity


################################################################################
#      Class for extracting software injections from an injection set          #
################################################################################

# For more information on contents of hdf5 file, the code that produced this file is:
# https://github.com/ligo-cbc/pycbc/blob/master/bin/hdfcoinc/pycbc_coinc_hdfinjfind

# !!! To properly get redshift, you should match the template_id to the xml file 
# that created the injection set.
# The xml file has the redshift, but the hdf5 file didn't store it. !!!
# m1 and m2 are the observer-frame masses not source-frame masses.
def process_injection_set(hdf5file):
    """Extract data about the injections from the pycbc injection file.
    Sort so that m1>m2.
    Divide into lists of found and missed injections.
    
    Parameters
    ----------
    hdf5file : string
    
    Returns
    -------
    found : dictionary 
    missed : dictionary
    time : float
    """
    f = h5py.File(hdf5file, 'r')
    
    # m1, m2, distance of all injections
    # This also includes injections when both detectors weren't in analysis time
    m1, m2 = f['injections/mass1'][:], f['injections/mass2'][:]
    s1x, s2x = f['injections/spin1x'][:], f['injections/spin2x'][:]
    s1y, s2y = f['injections/spin1y'][:], f['injections/spin2y'][:]
    s1z, s2z = f['injections/spin1z'][:], f['injections/spin2z'][:]
    
    # switch label of stars if m1<m2
    for i in range(len(m1)):
        if m1[i] < m2[i]:
            m1[i], m2[i] = m2[i], m1[i]
            s1x[i], s2x[i] = s2x[i], s1x[i]
            s1y[i], s2y[i] = s2y[i], s1y[i]
            s1z[i], s2z[i] = s2z[i], s1z[i]
    
    dist = f['injections/distance'][:]
    mchirp = (m1*m2)**(3./5.) / (m1+m2)**(1./5.)
    chirpdist = (mchirp/(1.4+1.4))**(-5./6.) * dist
    s1 = np.sqrt(s1x**2+s1y**2+s1z**2)
    s2 = np.sqrt(s2x**2+s2y**2+s2z**2)
    
    # Redshift isn't stored in hdf5 file, so 
    # get approximate redshift and source frame masses from Hubble's law.
    # !!! There are more accurate ways of getting redshift from distance. !!!
    H0 = 67.8
    ckms = 299792.458
    redshift = H0*dist/ckms
    m1source = m1/(1.0+redshift)
    m2source = m2/(1.0+redshift)
    
    # injections that were found in the analysis time (after vetoing time)
    found_i = f['found_after_vetoes/injection_index'][:]
    found_ifar = f['found_after_vetoes/ifar_exc'][:]
    found_stat = f['found_after_vetoes/stat'][:]
    
    # injections that were missed in the analysis time (after vetoing time)
    missed_i = f['missed/after_vetoes'][:]
    
    # analysis time (after vetoing time) in years
    time = f.attrs['foreground_time_exc'] / (365.25*86400)
    
    # return found and missed injections
    
    found = {'m1': m1[found_i], 
             'm2': m2[found_i], 
             's1x': s1x[found_i], 
             's2x': s2x[found_i], 
             's1y': s1y[found_i], 
             's2y': s2y[found_i], 
             's1z': s1z[found_i], 
             's2z': s2z[found_i], 
             's1': s1[found_i], 
             's2': s2[found_i], 
             'mchirp': mchirp[found_i], 
             'chirpdist': chirpdist[found_i], 
             'dist': dist[found_i],
             'redshift': redshift[found_i],
             'm1source': m1source[found_i],
             'm2source': m2source[found_i],
             'ifar': found_ifar, 
             'stat': found_stat}
    
    missed = {'m1': m1[missed_i], 
              'm2': m2[missed_i], 
              's1x': s1x[missed_i], 
              's2x': s2x[missed_i], 
              's1y': s1y[missed_i], 
              's2y': s2y[missed_i], 
              's1z': s1z[missed_i], 
              's2z': s2z[missed_i], 
              's1': s1[missed_i], 
              's2': s2[missed_i], 
              'mchirp': mchirp[missed_i], 
              'chirpdist': chirpdist[missed_i], 
              'dist': dist[missed_i],
              'redshift': redshift[missed_i],
              'm1source': m1source[missed_i],
              'm2source': m2source[missed_i],}
    
    return found, missed, time


class InjectionSet(object):
    """Found and missed software injections.
    
    Attributes
    ----------
    found
    missed
    time
    """
    
    def __init__(self, found, missed, time):
        """
        Parameters
        ----------
        found : dictionary
            Properties of the found injections.
        missed : dictionary
            Properties of the missed injections.
        time : float
            Analysis time after detchar vetoes (years).
        """
        self.found = copy.deepcopy(found)
        self.missed = copy.deepcopy(missed)
        self.time = time
        
    @classmethod
    def load(cls, hdf5file):
        """Load found and missed software injections from hdf5 file.
        
        Parameters
        ----------
        hdf5file : string
            File containing the found and missed injections.
        """
        found, missed, time = process_injection_set(hdf5file)
        return cls(found, missed, time)
    
    def get_all(self, key):
        """Concatenate found and missed injections for the parameter key.
        """
        return np.concatenate((self.found[key], self.missed[key]))
    
    def print_ranges(self):
        print('m1source: ({:.4f}, {:.4f})'.format(np.min(self.get_all('m1source')), np.max(self.get_all('m1source'))))
        print('m2source: ({:.4f}, {:.4f})'.format(np.min(self.get_all('m2source')), np.max(self.get_all('m2source'))))
        print('s1: ({:.4f}, {:.4f})'.format(np.min(self.get_all('s1')), np.max(self.get_all('s1'))))
        print('s2: ({:.4f}, {:.4f})'.format(np.min(self.get_all('s2')), np.max(self.get_all('s2'))))

    def bin_2d(self, key1, key2, range1, range2):
        """Create an InjectionSet object containing only injections in a 2d mass bin.
        
        Parameters
        ----------
        key1 : str
            Name of first parameter.
        key2 : str
            Name of second parameter.
        range1 : list [key1low, key1high]
        range2 : list [key2low, key2high]
        
        Returns
        -------
        bin2d : InjectionSet
        """
        # indices of found injections in the 2-d bin
        N = len(self.found[key1])
        k1 = self.found[key1]
        k2 = self.found[key2]
        f_bin_i = np.array([
            i for i in range(N) if 
            k1[i]>=range1[0] and k1[i]<=range1[1] and k2[i]>=range2[0] and k2[i]<=range2[1]
            ])
        
        # indices of missed injections in the 2-d bin
        N = len(self.missed[key1])
        k1 = self.missed[key1]
        k2 = self.missed[key2]
        m_bin_i = np.array([
            i for i in range(N) if 
            k1[i]>=range1[0] and k1[i]<=range1[1] and k2[i]>=range2[0] and k2[i]<=range2[1]
            ])
        
        # Copy the InjectionSet object
        # Then replace the found and missed injections with the subset in the mass bin
        bin2d = InjectionSet(self.found, self.missed, self.time)
        
        for key, val in bin2d.found.items():
            if len(m_bin_i) == 0:
                bin2d.found[key] = np.array([])
            else:
                bin2d.found[key] = val[f_bin_i]
        
        for key, val in bin2d.missed.items():
            if len(m_bin_i) == 0:
                bin2d.missed[key] = np.array([])
            else:
                bin2d.missed[key] = val[m_bin_i]
            
        return bin2d
    
    def sort_by_threshold(self, ifarthresh=None, statthresh=None):
        """Split up found and missed injections using a threshold for IFAR.
        """
        # Get indices of found injections above and below threshold
        if ifarthresh:
            ifar = self.found['ifar']
            above = ifar >= ifarthresh
            below = ifar < ifarthresh
        elif statthresh:
            stat = self.found['stat']
            above = stat >= statthresh
            below = stat < statthresh
        else:
            raise Exception, 'You must specify ifrathresh or statthresh.'
        
        # Copy the InjectionSet object
        # Then redivide into found and missed
        redivided = InjectionSet(self.found, self.missed, self.time)
        
        for key, val in redivided.missed.items():
            redivided.missed[key] = np.append(val, redivided.found[key][below])
            
        for key, val in redivided.found.items():
            redivided.found[key] = val[above]
        
        return redivided
    
    def reduce_samples(self, Nmax='all'):
        """Remove all but the first Nmax samples in the injection set.
        The ratio of found to missed injections will stay the same (+/- 1 injection).
        
        Parameters
        ----------
        Nmax : {'all', int}
            Number of samples to keep.
            
        Returns
        -------
        reduced : InjectionSet
        """
        if Nmax=='all':
            'Return a copy of the current InjectionSet.'
            return InjectionSet(self.found, self.missed, self.time)
        else:
            Nall = len(self.get_all('m1'))
            Nfound = len(self.found['m1'])
            Nmissed = len(self.missed['m1'])
            if Nmax>Nall: raise Exception, 'Nmax must be <= the number of injected waveforms Ninj.'
            
            # Reduced found and missed samples should have the same proportion as the old.
            Nfound_red = int(np.floor(Nmax*Nfound/float(Nall)))
            Nmissed_red = Nmax-Nfound_red
            
            # Copy the InjectionSet object
            # Then redivide into found and missed
            reduced = InjectionSet(self.found, self.missed, self.time)
            
            for key, val in reduced.found.items():
                reduced.found[key] = reduced.found[key][:Nfound_red]
                
            for key, val in reduced.missed.items():
                reduced.missed[key] = reduced.missed[key][:Nmissed_red]
            
            return reduced
        
    def sensitive_volume(self, ifarthresh=None, statthresh=None, bins=50):
        """Calculate the sensitive volume of the detector.
        """
        redivided = self.sort_by_threshold(ifarthresh=ifarthresh, statthresh=statthresh)
        
        f_dist = redivided.found['dist']
        m_dist_full = redivided.missed['dist']
        vol, vol_err = pycbc.sensitivity.volume_binned_pylal(f_dist, m_dist_full, bins=bins)
    
        return vol, vol_err


################################################################################
#   2 Methods for adding up the VT from multiple injection sets.               #
#   Only the Monte Carlo method should be used with Jolien's injection sets.   #
################################################################################

# !!!! Warning !!!! This will not work with Jolien's injection sets 
# which cuts low SNR sources before injecting them into pycbc/gstlal.
# Use the add_vt_of_injection_sets_monte_carlo function instead.
# It will work with any other injection set that doesn't make cuts first.
def add_vt_of_injection_sets(inj_list, ifarthresh=None, statthresh=None, bins=50, Nmax='all'):
    """Calculate vt for a list of injections.
    Errors are calculated by treating v as gaussian random variable with expectation value v and variance verr^2.
    t is a constant, so Var(VTtotal) = sum((verr_i*t_i)^2)
    
    Parameters
    ----------
    Nmax : {'all', int}
        Number of samples to keep.
        
    Returns
    -------
    vt : float
        sum(v_i*t_i)
    vterr : float
        sqrt(sum((verr_i*t_i)^2))
    """
    vt = 0.0
    vterrsq = 0.0
    for i in range(len(inj_list)):
        inj = inj_list[i]
        # Reduce samples if requested
        if Nmax is not 'all': 
            inj = inj.reduce_samples(Nmax)
        # Calculate VT for each InjectionSet
        v, verr = inj.sensitive_volume(ifarthresh=ifarthresh, statthresh=statthresh, bins=bins)
        t = inj_list[i].time
        vt += v*t
        vterrsq += (verr*t)**2

    return vt, np.sqrt(vterrsq)


def add_vt_of_injection_sets_monte_carlo(inj_list, Ntotal, VT_all_inj, ifarthresh=None, statthresh=None):
    """Calculate vt for a list of injections.
    
    Parameters
    ----------
    Ntotal : int
        Total number of samples which may be larger than the samples in the hdf5file
        since not all sampled events are actually injected.
    VT_all_inj : float
        The VT containing the Ntotal injected samples.
    """
    Nfound = 0
    for i in range(len(inj_list)):
        inj = inj_list[i]
        
        # Resort each injection set into found and missed lists
        redivided = inj.sort_by_threshold(ifarthresh=ifarthresh, statthresh=statthresh)
        
        # Sensitive VT estimate
        Nfound += len(redivided.found['m1'])
    
    # Sensitive VT estimate
    p = float(Nfound) / float(Ntotal)
    VT = p * VT_all_inj
    
    # Monte Carlo counting error in sensitive VT estimate
    VT_err = VT_all_inj * np.sqrt(p*(1-p)/Ntotal)
    
    return VT, VT_err


################################################################################
#           Poisson Likelihood, priors on Lambda and VT,                       #
#           class for calculating rate upper limit.                            #
################################################################################

def poisson0(r, vt):
    """Likelihood function p(d|r, vt)
    when we are confident there are n=0 detections.
    """
    return np.exp(-r*vt)


def prior_lambda_uniform(r, vt):
    """Uniform/no prior on Lambda=r*vt.
    """
    return 1.0


def prior_lambda_jeffreys(r, vt):
    """Jeffreys prior on Lambda=r*vt.
    """
    return 1.0/np.sqrt(r*vt)


def prior_vt_uniform(vt, vtbar, f):
    """Unnormalized uniform prior on vt 'geometrically centered' on vtbar 
    with bounds [vtbar*(1-f), vtbar*(1+f)].
    """
    if vt > vtbar*(1.0-f) and vt < vtbar*(1.0+f):
        return 1.0
    else:
        return 0.0


def prior_vt_lognormal(vt, vtbar, f):
    """Unnormalized log-normal prior on vt 'geometrically centered' on vtbar 
    with fractional uncertainty f.
    """
    return (1.0/vt) * np.exp( -np.log(vt/vtbar)**2 / (2.0*f**2) )


class RateModel(object):
    def __init__(self, like_lambda, prior_lambda, prior_vt, vtbar, f, 
                 rlow=None, rhigh=None, vtlow=None, vthigh=None):
        """Model for the rate of inspiral events.
        
        Attributes
        ----------
        like_lambda : func(r, vt)
            Likelihood function p(d|Lambda).
        prior_lambda : func(r, vt)
            Prior p(Lambda) on Lambda.
        prior_vt : func(vt, vtbar, f)
            Prior on VT.
        vtbar : float
            Measured value of VT.
        f : float
            Fractional uncertainty in VT.
        """
        # Order of magnitude estimate of the rate
        self.r_estimate = 1.0 / vtbar
        
        # Default bounds of integration.
        # Don't use rlow=0 in case p(r|d) blows up at r=0 (as with Jeffreys prior).
        if rlow is None: rlow = 1.0e-9*self.r_estimate
        #if rhigh is None: rhigh = np.inf
        if rhigh is None: rhigh = 1.0e3*self.r_estimate
        if vtlow is None: vtlow = 1.0e-4*vtbar
        if vthigh is None: vthigh = 1.0e4*vtbar 
        self.rlow, self.rhigh, self.vtlow, self.vthigh = rlow, rhigh, vtlow, vthigh
        
        self.like = like_lambda
        self.prior_lambda = prior_lambda
        self.prior_vt = prior_vt
        self.vtbar = vtbar
        self.f = f
        self.norm_r = self.posterior_r_norm()
        
    def posterior_r_vt(self, r, vt):
        """Unnormalized posterior p(r, vt|d).
        """
        return self.prior_vt(vt, self.vtbar, self.f) * self.prior_lambda(r, vt) * self.like(r, vt)
    
    def posterior_r(self, r):
        """Unnormalized posterior p(r|d). 
        (p(r, vt|d) marginalized over vt.) 
        """
        # Break points around where function changes rapidly as a function of vt
        points = (self.vtbar*(1.0-self.f), self.vtbar*(1.0+self.f))
        
        # Numerical integrator requires integral to be over first argument
        integrand = lambda vt, r : self.posterior_r_vt(r, vt)
        return integrate.quad(integrand, self.vtlow, self.vthigh, args=(r), points=points)[0]
    
    def posterior_r_norm(self):
        """Calculate the normalization constant for the marginalized posterior p(r|d).
        """
        return integrate.quad(self.posterior_r, self.rlow, self.rhigh)[0]
    
    def posterior_r_normalized(self, r):
        """Normalized posterior p(r|d). (
        p(r, vt|d) marginalized over vt.) 
        """
        return self.posterior_r(r) / self.norm_r
    
    def rate_upper_limit(self, conf):
        """Find upper limit with confidence conf.
        
        Solves the differential equation dy/dr = p(r|d)
        to evaluate the integral y(R) = int_0^R p(r|d)dr for all values of R.
        
        Parameters
        ----------
        conf : float
            Confidence.
        """    
        # Differential equation for confidence
        def func(y, r):
            return self.posterior_r_normalized(r)
        
        # Initial condition
        y0 = 0.0
        
        r0 = self.rlow
        rmax = 10.0*self.r_estimate
        nsamp = 10000
        rs = np.linspace(r0, rmax, nsamp)
    
        confidence = integrate.odeint(func, y0, rs).T[0]
        
        Rofconf = interpolate.interp1d(confidence, rs)
        return float(Rofconf(conf))  


################################################################################
#                   Analytic results for comparison                            #
################################################################################

#########    Rate posteriors p(R|d) marginalized over priors on VT.   ##########

def poisson_uniformr_deltavt(rate, vtbar):
    """Poisson process with n=0. p(R|d).
    """
    return vtbar*np.exp(-rate*vtbar)

def poisson_uniformr_uniformvt(rate, vtbar, f):
    num = np.exp(-vtbar*rate*(1.0-f)) - np.exp(-vtbar*rate*(1.0+f))
    den = np.log((1.0+f)/(1.0-f))*rate
    return num/den

def poisson_jeffreysr_deltavt(rate, vtbar):
    return np.sqrt(vtbar/(np.pi*rate)) * np.exp(-rate*vtbar)

def poisson_jeffreysr_uniformvt(rate, vtbar, f):
    num = -special.erf(np.sqrt(vtbar*rate*(1.0-f))) + special.erf(np.sqrt(vtbar*rate*(1.0+f)))
    den = np.log((1.0+f)/(1.0-f))*rate
    return num/den

######## Rate upper limits in the limit of 0 uncertainty in VT (f->0). #########

def rate_upper_limit_uniform(conf, vtbar):
    """Analytic upper limit on rate for uniform prior on R and zero uncertainty in VT.
    """
    return -np.log(1-conf)/vtbar

def rate_upper_limit_jeffreys(conf, vtbar):
    """Analytic upper limit on rate for Jeffreys prior on R and zero uncertainty in VT.
    """
    return special.erfinv(conf)**2/vtbar

