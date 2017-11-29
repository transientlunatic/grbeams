#!/usr/bin/env python
# Copyright (C) 2016-2017 Daniel Williams <daniel.williams@ligo.org>
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
This script is designed to reproduce the surface plots shown in
figures 7 and 8 of Williams et al. 2017, using data produced by the
surface_data.py script.

"""

import scipy.ndimage
from matplotlib import colors, ticker, cm
from astropy import constants as c
from astropy import units as u

width = 3.487 #* 2
height = width / 1.618

data = np.loadtxt('upper.dat')
matplotlib.rcParams.update({'font.size': 6})

scenarios = {
    #"A": 4*0.45,
    #"B": 2*1.3,
    #"C": 6.5/0.75,
    "2019+": 20,
    "2022+": 40
            }

#height, width= 20, 10
f, ax = plt.subplots(1,1, sharex=True, figsize = (width, height))



ax.set_xlim([0, 45]);

vth = 4./3 * np.pi #* (c.c * 1*u.year).to(u.megaparsec)
for scenario in scenarios.iteritems():
    ax.vlines(scenario[1], 0, 19, color='k', alpha=0.5, lw=2)
    x_bounds = ax.get_xlim()
    ax.annotate(s=scenario[0], xy =(((scenario[1]-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), xycoords='axes fraction', verticalalignment='right', rotation = 0)

    
sigma = 0.7
from scipy.ndimage.filters import gaussian_filter
data = gaussian_filter(data, sigma)

consf = ax.contourf(np.linspace(vth*(2.26*10)**3/1e6, vth*(2.26*100)**3/1e6, data.shape[1]), np.linspace(0, 19, data.shape[0]), 
             data, 
             levels = np.linspace(0, 90, 10), alpha = 0.4, cmap = "magma_r", lw=0
            )



ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', linewidth=1.0, color="#DDDDDD")
ax.grid(b=True, which='minor', linewidth=0.5)
cb=f.colorbar(consf)
ax.set_xlabel(r"Search 4-volume, $VT\, [\times10^6\, \rm{Mpc}^3 \rm{yr}]$")
ax.set_ylabel(r"GW BNS event rate [${\rm yr}^{-1}$]")

cb.set_label('Beaming angle upper limit [$^{\circ}$]')
f.subplots_adjust(0.1, 0.15, 1, .95, wspace=0.05)
f.savefig("volume_v_nevents.pdf")

import scipy.ndimage
from matplotlib import colors, ticker, cm
from astropy import constants as c
from astropy import units as u

data = np.loadtxt('lower.dat')


#height, width= 20, 10
f, ax = plt.subplots(1,1, sharex=True, figsize = (width, height))



ax.set_xlim([0, 45]);

vth = 4./3 * np.pi #* (c.c * 1*u.year).to(u.megaparsec)
for scenario in scenarios.iteritems():
    ax.vlines(scenario[1], 0, 19, color='k', alpha=0.5, lw=2)
    x_bounds = ax.get_xlim()
    ax.annotate(s=scenario[0], xy =(((scenario[1]-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), xycoords='axes fraction', verticalalignment='right', rotation = 0)

    
sigma = 0.7
from scipy.ndimage.filters import gaussian_filter
data = gaussian_filter(data, sigma)
    
consf = ax.contourf(np.linspace(vth*(2.26*10)**3/1e6, vth*(2.26*100)**3/1e6, data.shape[1]), np.linspace(0, 19, data.shape[0]), 
             data, 
             levels = np.linspace(0, 30, 11), alpha = 0.4, cmap = "magma", lw=0
            )

#cons = ax.contour(np.linspace(vth*10**3, vth*400**3, data.shape[1]), np.linspace(0, 10, data.shape[0]), 
#             data, 
#             levels = [0, 1, 2, 3, 4, 5, 10, 20], colors='maroon', lw=1, cw=1
#            )

#cons = ax.contour(np.log10(data))

#ax.clabel(cons, inline=1, fontsize=8, fmt='%1.0f')

ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', linewidth=1.0, color="#DDDDDD")
ax.grid(b=True, which='minor', linewidth=0.5)
cb=f.colorbar(consf)
ax.set_xlabel(r"Search 4-volume, $VT\, [\times10^6\, \rm{Mpc}^3 \rm{yr}]$")
ax.set_ylabel(r"GW BNS event rate [${\rm yr}^{-1}$]")

cb.set_label('Beaming angle lower limit [$^{\circ}$]')
f.subplots_adjust(0.1, 0.15, 1, .95, wspace=0.05)
f.savefig("../final_paper/volume_v_nevents_lower.pdf")
