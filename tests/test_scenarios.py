#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_scenarios
----------------------------------

Tests for `scenarios` module.
"""

import unittest
import numpy as np
import astropy.units as u
from grbeams import scenarios


class TestScenarios(unittest.TestCase):

    def setUp(self):
        self.rateunits = u.megaparsec**-3*u.year**-1
        # Create a BNS distribution which is uniform
        rates = np.linspace(0, 100, 100)*self.rateunits
        pdf = np.ones(len(rates)) * (1.0/100) * self.rateunits**-1
        self.bnsuniform = scenarios.BNSDistribution(rates, pdf)
        # Create an observing scenario with the same BNS distribution
        self.uniformscenario = scenarios.Scenario(self.bnsuniform)
        self.grb_rate = 3*self.rateunits

    def test_bns_negative_rate(self):
        # A negative rate has no physical meaning, and should return a probability of zero.
        self.assertEqual(self.bnsuniform.pdf(-1*self.rateunits), 0)

    def test_bns_excessive_rate(self):
        # Return zero if the rate is higher than the pdf data provided.
        self.assertEqual(self.bnsuniform.pdf(1000*self.rateunits), 0)

    def test_bns_in_range(self):
        # Return a non-zero probability from the distro in range
        self.assertGreater(self.bnsuniform.pdf(50*self.rateunits), 0)

    def test_scenario_bns_prior(self):
        # Test that the scenario correctly returns the BNS Prior
        self.assertEqual(self.uniformscenario.comp_bns_rate_pdf(40*self.rateunits), self.bnsuniform.pdf(40*self.rateunits))

    def test_scenario_cbc_rate_negative(self):
        self.assertEqual(self.uniformscenario.cbc_rate_from_theta(theta=10, efficiency=-0.5, grb_rate = self.grb_rate), 0)
                         
    def test_scenario_cbc_rate_gt1(self):
        self.assertEqual(self.uniformscenario.cbc_rate_from_theta(theta=10, efficiency=1.5, grb_rate = self.grb_rate), 0)
                                                  
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
