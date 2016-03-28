#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_distributions
----------------------------------

Tests for `distributions` module.
"""

import unittest
import numpy as np
from grbeams import distributions


class TestDistributions(unittest.TestCase):

    def setUp(self):
        # Create a delta distribution with its spike at x=0.5
        self.deltadist = distributions.DeltaDistribution(x=0.5)
        # Create a Uniform distribution over (0.5, 0.7)
        self.uniformdist = distributions.UniformDistribution(range=(0.5, 0.7))
        # Create a Jeffrey's distribution
        self.jeffrey = distributions.JeffreyDistribution()

    def test_delta_on_value(self):
        # Test the delta distribution at the point of the spike.
        self.assertEqual(self.deltadist.pdf(0.5), 1.0)

    def test_delta_off_value(self):
        # Test he delta distribution off the point of the spike.
        self.assertEqual(self.deltadist.pdf(0.1), 0)

    def test_delta_integral(self):
        # Test that the whole distribution integrates to give 1
        x = np.linspace(-1, 1, 1001)
        probs = [self.deltadist.pdf(place) for place in x]
        self.assertEqual(sum(probs), 1)

    def test_uniform_in_range(self):
        # Test that the probability is non-zero inside the distribution's range
        self.assertGreater(self.uniformdist.pdf(0.6), 0)

    def test_uniform_out_range(self):
        # Test that the probability is zero outside the range
        self.assertEqual(self.uniformdist.pdf(0.1, 0),0)

    def test_uniform_integral(self):
        # Check that the distribution integrates to 1
        
        x = np.linspace(-1, 1, 1001)
        probs = [self.deltadist.pdf(place) for place in x]
        self.assertEqual(sum(probs), 1)

    def test_uniform_in_range(self):
        # Test that the probability is non-zero inside the distribution's range
        self.assertGreater(self.uniformdist.pdf(0.6), 0)

    def test_uniform_out_range(self):
        # Test that the probability is zero outside the range
        self.assertEqual(self.uniformdist.pdf(0.1),0)

    def test_uniform_integral(self):
        # Check that the distribution integrates to 1
        x = np.linspace(-1, 1, 1001)
        probs = np.array([self.uniformdist.pdf(place) for place in x])
        self.assertAlmostEqual(np.trapz(probs, x=x), 1)

    def test_uniform_edge(self):
        # Check that the lower edge is included, but the upper edge is not
        self.assertGreater(self.uniformdist.pdf(0.5), 0)
        self.assertEqual(self.uniformdist.pdf(0.7), 0)

    def test_jeffrey_in_range(self):
        # Check that the Jeffrey distribution has a value > 0 in its range
        self.assertGreater(self.jeffrey.pdf(0.5),0)

    def test_jeffrey_out_range(self):
        # Check that the Jeffrey distribution has a value > 0 in its range
        self.assertEqual(self.jeffrey.pdf(-0.5),0)
        self.assertEqual(self.jeffrey.pdf(1.5),0)

    def test_jeffrey_integral(self):
        # Check that the distribution integrates to inf (it's non-normalisable)
        x = np.linspace(0, 1, 1001)
        probs = np.array([self.jeffrey.pdf(place) for place in x])
        self.assertAlmostEqual(np.trapz(probs, x=x), np.inf)
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
