import unittest
import numpy as np
import astropy.units as u
from grbeams import scenarios
from grbeams.distributions import *
from grbeams.beamingangle import BeamingAnglePosterior

class TestBeaming(unittest.TestCase):

    def setUp(self):
        self.rateunits = u.megaparsec**-3*u.year**-1
        # Create a BNS distribution which is uniform
        rates = np.linspace(0, 100, 100)*self.rateunits
        pdf = np.ones(len(rates)) * (1.0/100) * self.rateunits**-1
        self.bnsuniform = scenarios.BNSDistribution(rates, pdf)
        # Create an observing scenario with the same BNS distribution
        self.uniformscenario = scenarios.Scenario(self.bnsuniform)
        self.grb_rate = 3*self.rateunits
        self.beaming = BeamingAnglePosterior(self.uniformscenario, 
                                             DeltaDistribution(0.3), 
                                             grb_rate=self.grb_rate )

    def test_efficiency_prior_negative(self):
        # Test the response to a negative efficiency
        # This should return a probability of zero
        self.assertEqual(self.beaming.comp_efficiency_prob(-0.5), 0)

    def test_efficiency_prior_gt1(self):
        # Test the response to an efficiency greater than 1
        self.assertEqual(self.beaming.comp_efficiency_prob(1.5), 0)

    def test_efficiency_prior_in_range(self):
        # Test the response to an efficiency between 0 and 1
        self.assertGreater(self.beaming.comp_efficiency_prob(0.3),0)

    ###
    def test_jacobian_at_zero(self):
        # Test that the Jacobian behaves as expected, which is to give
        # a nan value when the theta angle is zero.
        self.assertGreater(self.beaming.compute_jacobian(efficiency=0.3,
                                                         theta = 0), np.nan)

    ####

    def test_logpdf_theta_outrange(self):
        # Test the log pdf of the beaming angle outside the 
        # prior range, should return 0
        self.assertEqual(self.beaming.logpdf(-100, 0.5), -np.inf)
        self.assertEqual(self.beaming.logpdf(+100, 0.5), -np.inf)

    def test_logpdf_theta_inrange(self):
        # Test the logpdf of the beaming angle inside the 
        # prior range, which should return a positive number
        self.assertGreater(self.beaming.logpdf(theta = 45, efficiency = 0.5), 0)
        

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()

