#!/usr/bin/env python
# Copyright (C) 2015-2016 James Clark <james.clark@physics.gatech.edu>
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

import sys
import numpy as np
from matplotlib import pyplot as pl


resultsfiles = sys.argv[1].split(',')

f, ax = pl.subplots()
labels=[r"\delta(1)", r"U(0,1)", r"\beta(0,1)"]

linestyles=['-', '--', ':']
for r,result in enumerate(resultsfiles):

    data = np.load(result)

    if r==2: linewidth=2
    else: linewidth=1
    ax.plot(data['thetaAxis'], data['thetaPos'], linestyle=linestyles[r],
            color='k', label=r'$p(\theta|I)=%s$'%labels[r], linewidth=linewidth)
    ax.axvline(data['thetaNinety'], color='k', linestyle=linestyles[r])#, label=r'$90\%$ U.L')

ax.legend()
ax.minorticks_on()

ax.set_xlabel(r'Jet Angle, $\theta$ [deg]')
ax.set_ylabel(r'$p(\theta|D,I)$')
f.tight_layout()

pl.show()
sys.exit()

savefig('jet_angle_posterior_s6UL.eps'%outputname)
savefig('jet_angle_posterior_s6UL.png'%outputname)
savefig('jet_angle_posterior_s6UL.pdf'%outputname)



