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
import numpy as np
from scipy import stats
from matplotlib import pyplot as pl

eff = np.arange(0,1,0.001)
prior_flat = np.ones(len(eff))
prior_jeff_pdf = stats.beta(0.5,0.5)
prior_jeff = prior_jeff_pdf.pdf(eff)

f, ax = pl.subplots()
ax.semilogy(eff, prior_flat, label='uniform', color='k')
ax.semilogy(eff, prior_jeff, label=r'Jeffreys: $\beta(1/2,1/2)$', color='k',
        linestyle='--')
ax.set_xlabel('GRB Efficiency, $\epsilon$')
ax.set_ylabel('$p(\epsilon|I)$')
ax.legend()
ax.minorticks_on()
f.tight_layout()

pl.show()






