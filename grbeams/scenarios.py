from .distributions import Distribution
import astropy.units as u
import numpy as np

class BNSDistribution():
    def __init__(self, rate_data, rate_pdf):
        """
        Represent a Binary Neutron Star distribution, produced from interpolated arrays.

        Parameters
        ----------
        rate_data : ndarray
           The rates at which a PDF has been calculated.
        rate_pdf : ndarray
           The pdf at each computed rate value.
        """
        self.rates = rate_data.to(u.megaparsec**(-3)*u.year**(-1))
        self.pdf_data = rate_pdf#.to(u.megaparsec**(3)*u.year**(1))

    def pdf(self, rate):
        """
        Return the probability density at a given rate.

        Parameters
        ----------
        rate : float
           The rate at which the PDF should be evaluated.
        
        Returns
        -------
        probability : float
           The probability density of that rate.
        """
        try:
            rate = rate.to(u.megaparsec**(-3)*u.year**(-1))
        except:
            pass
        if rate < 0 : return 0
        if rate > max(self.rates): return 0
        try:
            return np.interp(rate.value, self.rates.value, self.pdf_data)
        except:
            return np.interp(rate, self.rates, self.pdf_data)

class Scenario():
    """
    Represents an observing scenario.
    """
    
    def __init__(self, bns_prior): # rate_posterior_file, upper_limit=1e-4):
        """
        A generic scenario class to contain the information 
        about the observing scenario.
    
        Parameters
        ----------
        rate_posterior_file : str
            The filepath to the file containing the rates posterior.
            
        upper_limit : float
            The upper limit on the posterior from the loudest event.
        """
        #posterior_data = np.loadtxt(rate_posterior_file)

        #bns_rate = self.posterior_data[:,1] *u.megaparsec**(-3)
        #bns_rate_pdf = self.posterior_data[:,0]

        self.bns_prior = bns_prior
        
        # Should probably compute this directly
        #self.upper_limit = upper_limit

    def comp_bns_rate_pdf(self, rate):
        return self.bns_prior.pdf(rate)
        
    def comp_grb_rate(self, efficiency, theta, bns_rate):
        """
        Computes the GRB rate:
        Rgrb = epsilon*(1-cos(theta))*Rbns
        """
        return efficiency*(1-np.cos(theta/180.0 * np.pi))*bns_rate

    def cbc_rate_from_theta(self, grb_rate, theta, efficiency):
        """
        Returns Rcbc = Rgrb / (1-cos(theta))
        """
        if efficiency < 0 : return 0
        if efficiency > 1 : return 0
        return grb_rate / ( efficiency*(1.-np.cos(theta * np.pi / 180)) )
