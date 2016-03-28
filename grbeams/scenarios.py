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
        self.pdf_data = rate_pdf

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
        rate = rate.to(u.megaparsec**(-3)*u.year**(-1))
        if rate < 0 : return 0
        if rate > max(self.rates): return 0
        return np.interp(rate.value, self.rates.value, self.pdf_data)

# class Scenario():
#     """
#     Represents an observing scenario.
#     """
    
#     def __init__(self, rate_posterior_file, upper_limit=1e-4):
#         """
#         A generic scenario class to contain the information 
#         about the observing scenario.
    
#         Parameters
#         ----------
#         rate_posterior_file : str
#             The filepath to the file containing the rates posterior.
            
#         upper_limit : float
#             The upper limit on the posterior from the loudest event.
#         """
#         posterior_data = np.loadtxt(rate_posterior_file)

#         self.bns_rate = self.posterior_data[:,1] *u.megaparsec**(-3)
#         self.bns_rate_pdf = self.posterior_data[:,0]
        
#         # Should probably compute this directly
#         self.upper_limit = upper_limit

#     def plot_posterior(self):
#         self.compute_posteriors()
#         fig, ax = plt.subplots(1)
#         ax.plot(self.bns_rate, self.bns_rate_pdf, label="Posterior")
        
#     def comp_bns_rate_pdf(self, bns_rate):
#         pdf_value = 
#         return pdf_value
    
#     from scipy.misc import factorial
#     def rate_fit(x,C,T,b,n):
#         return C*T*( (x+b)*T )**n * np.exp(-(x+b)*T) / factorial(n)

#     def fit_rate(x,y):
#         popt,_ = optimize.curve_fit(rate_fit, x, y)
#         return popt

#     def comp_grb_rate(self, efficiency, theta, bns_rate):
#         """
#         Computes the GRB rate:
#         Rgrb = epsilon*(1-cos(theta))*Rbns
#         """
#         return efficiency*(1-np.cos(theta/180.0 * np.pi))*bns_rate

#     def cbc_rate_from_theta(self, grb_rate,theta,efficiency):
#         """
#         Returns Rcbc = Rgrb / (1-cos(theta))
#         """
#         return grb_rate / ( efficiency*(1.-np.cos(theta * np.pi / 180)) )
