import numpy as np
import matplotlib.pyplot as plt
import emcee
import astropy.units as u
from grbeams.distributions import *
from grbeams.scenarios import *


from sklearn.neighbors.kde import KernelDensity
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    
    N = np.trapz(np.exp(log_pdf), x_grid)

    return np.exp(log_pdf)/N

class BeamingAnglePosterior(Distribution):

    def __init__(self, 
                 observing_scenario, 
                 efficiency_prior=DeltaDistribution(1.0),#'delta,1.0',
                 grb_rate=3 / u.gigaparsec**3 / u.year):
        """
        The posterior probability distribution on the beaming angle, theta.
        
        Parameters
        ----------
        observing_scenario : Scenario object
            The observing scenario, e.g. O1.
        efficiency_prior : str
            The shape of the prior on epsilon, the efficiency at which
            BNS mergers produce sGRBs.
        grb_rate : 
            The rate of GRBs in the galaxy.
        """

        self.efficiency_prior = efficiency_prior

        # --- Prior Setup
        
        self.theta_range = np.arange(0.1, 90.0, 0.01)
        self.efficiency_range = efficiency_prior.range
        
        # --- Astro configuration
        self.grb_rate = grb_rate
        self.scenario = observing_scenario

    def characterise_dist(self, x,y,alpha):
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

    def get_theta_pdf_kde(self, bandwidth=1.0):

        self.theta_grid  = self.theta_range #np.arange(self.theta_range[0], self.theta_range[1], 0.01)
        self.theta_pdf_kde  = kde_sklearn(x=self.theta_samples,
                                          x_grid=self.theta_grid, 
                                            bandwidth=bandwidth, 
                                            algorithm='kd_tree') 
        self.theta_bounds, self.theta_median, self.theta_posmax = \
                self.characterise_dist(self.theta_grid, self.theta_pdf_kde, 0.95)

    def sample_theta_posterior(self, nburnin=100, nsamp=500, nwalkers=100):
        """
        Use emcee ensemble sampler to draw samples from the ndim parameter space
        comprised of (theta, efficiency, delta_theta, ...) etc

        The dimensionality is determined in __init__ based on the priors used

        The probability function used is the comp_theta_prob() method
        """

        theta0 = (max(self.theta_range)-min(self.theta_range)) * np.random.rand(nwalkers)
        p0 = theta0.reshape((nwalkers, 1))
        
        if self.efficiency_prior.ndim==2:
            
            efficiency0 = (max(self.efficiency_prior.range)-min(self.efficiency_prior.range)) * np.random.rand(nwalkers)
            
            p0 = np.transpose(np.array([theta0, efficiency0]))
            
        # Inititalize sampler
        if self.efficiency_prior.name=='delta':
            print "Delta Distro", self.efficiency_prior.ndim, 
            self.sampler = emcee.EnsembleSampler(nwalkers, self.efficiency_prior.ndim,
                    self.logpdf, args=[self.efficiency_prior.x])
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, self.efficiency_prior.ndim, 
                                                 self.logpdf_nparam)
        
        # Burn-in
        
        pos, prob, state = self.sampler.run_mcmc(p0, nburnin)
        self.sampler.reset()

        # Draw samples
        self.sampler.run_mcmc(pos, nsamp)

        # 1D array with samples for convenience
        if self.efficiency_prior.ndim==1:
            self.theta_samples = np.concatenate(self.sampler.flatchain)
        else:
            self.theta_samples = self.sampler.flatchain[:,0]
            self.efficiency_samples = self.sampler.flatchain[:,1]


        # Create bppu posterior instance for easy conf intervals and
        # characterisation
        #self.theta_pos = bppu.PosteriorOneDPDF('theta', self.theta_samples,
        #        injected_value=self.sim_theta)
        
        return self.sampler.flatchain

    def logpdf_nparam(self, x, fixed_args=None):
        #print x
        #print self.logpdf(theta=x[0], efficiency=x[1])
        return self.logpdf(theta=x[0], efficiency=x[1])

            
    def logpdf(self,theta,efficiency):
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
        
        if theta <= min(self.theta_range): return -np.inf
        if theta > max(self.theta_range): return -np.inf
        # Get BNS rate from theta, efficiency
        bns_rate = self.scenario.cbc_rate_from_theta(self.grb_rate, theta, efficiency)
        # Get value of rate posterior at this rate
        bns_rate_pdf = self.scenario.comp_bns_rate_pdf(bns_rate)
        # Compute jacobian
        jacobian = self.compute_jacobian(efficiency,theta).value
        theta_prob = bns_rate_pdf \
                     + np.log(jacobian) \
                     + np.log(self.comp_efficiency_prob(efficiency))
        if np.isnan(theta_prob ):
            print bns_rate, bns_rate_pdf, jacobian, self.comp_efficiency_prob(efficiency)
        return theta_prob
    
    def compute_jacobian(self,efficiency,theta):
        """
        Compute the Jacboian for the transformation from rate to angle
        """

        denom=efficiency*(np.cos(theta * np.pi/180)-1)
        output =  abs(2.0*self.grb_rate * np.sin(theta * np.pi / 180.0) /
                (denom*denom) )
        if np.isinf(output):
            print efficiency, theta, denom, output
            return 0
        else:
            return output

    def comp_efficiency_prob(self,efficiency):
        """
        Prior on the BNS->GRB efficiency
        """
        return self.efficiency_prior.pdf(efficiency)
