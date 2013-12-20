#!/usr/bin/env python
# Copyright (C) 2012-2013 James Clark <clark@physics.umass.edu>
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

from pylab import *

fig_width_pt = 245.26653  # Get this from LaTeX using \showthe\columnwidth
#fig_width_pt=600.
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
rcParams.update(
		{'axes.labelsize': 10,
        'text.fontsize':   10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'figure.figsize': fig_size,
        'font.family': "serif",
        'font.serif': ["Times"],
        'savefig.dpi': 200,
        'xtick.major.size':8,
        'xtick.minor.size':4,
        'ytick.major.size':8,
        'ytick.minor.size':4
        })  

def cbcRatePosteriorNull(eps,rateAxis):
    """
    Returns the value of the posterior on the cbc Rate for Lambda=0
    """
    unnormed=eps*exp(-rateAxis*eps)
    N=trapz(unnormed,rateAxis)
    return unnormed/N

def rateFromTheta(theta,epsilon,grbRate):
    """
    Returns Rcbc = Rgrb / (1-cos(theta))
    """
    return grbRate / ( epsilon*(1.-cos(theta * pi / 180)) )

def jacobian(grbRate,epsilon,theta):
    """
    Compute the Jacboian for the transformation from rate to angle
    """
    denom=epsilon*(1-cos(theta * pi/180))
    return abs(2*grbRate * sin(theta * pi / 180) / denom**2)

def computeEpsilon(R90=1.3e-4,rateAxis=linspace(1e-8,1e-3,1000)):
    """
    Find the value of epsilon in the rate posterior such that the 90% upper
    limit on the rate is equal to that in the S6/VSR2,3 Lowmass search for null
    detection
    """
    epsilons=arange(17000,18000,10)
    allR90s=[]
    for eps in epsilons:
        pRcbc=cbcRatePosteriorNull(eps,rateAxis)
        alphas=[]
        for R in rateAxis:
            thisXaxis=rateAxis[rateAxis<=R]
            thisYaxis=pRcbc[rateAxis<=R]
            alphas.append(trapz(y=thisYaxis, x=thisXaxis, dx=diff(rateAxis)[0]))

        thisR90=interp(0.9,alphas,rateAxis)
        allR90s.append(thisR90)
    
    return interp(R90,allR90s[::-1],epsilons[::-1])

rateAxis=linspace(1e-8,1e-3,1000)

# found this by iterating through eps vals until i got the 90% rate
# upper limit from the S6 lowmass paper
#eps=computeEpsilon(rateAxis=rateAxis) 

# Here's the simple analytic expression:
eps=-1*log(1-0.9)/1.3e-4

cbcRatePos=cbcRatePosteriorNull(eps,rateAxis)
alphas=[]
for bin in rateAxis:
    alphas.append(trapz(y=cbcRatePos[rateAxis<bin], x=rateAxis[rateAxis<bin], dx=diff(rateAxis)[0]))
rateNinety=interp(0.9,alphas,rateAxis)
print '%d %0.3e'%(eps,rateNinety)

# Transform to jet angle:
#thetaAxis=arange(0.1,1,0.01)
thetaAxis=arange(0.01,90,0.01)
grbRate=10./ 1e9  # 10 is Gpc^-3...
thetaPos=[]

# Loop through jet angles:
# 1) compute jacobian for transformation
# 2) find the Rcbc corresponding to this Rgrb and theta (i.e., get p(theta) in
# terms of Rcbc).  Then multiply by Jacobian to get correct density
# 3) interpolate the Rcbc posterior to this Rcbc

# XXX Here's where epsilon comes in
epsilon=1.
for theta in thetaAxis:

    dRbydTheta=jacobian(grbRate,epsilon,theta)    
    cbcRate=rateFromTheta(theta,epsilon,grbRate)

    thetaPos.append(interp(cbcRate,rateAxis,cbcRatePos)*dRbydTheta)

# Ensure normalisation to unity:
N=trapz(thetaPos,thetaAxis)
thetaPos/=trapz(thetaPos,thetaAxis)

# Now integrate FROM 90
alphas=[]
for theta in thetaAxis:
    #alphas.append(1-trapz(thetaPos[thetaAxis>theta],thetaAxis[thetaAxis>theta]))
    alphas.append(trapz(thetaPos[thetaAxis<theta],thetaAxis[thetaAxis<theta]))
thetaNinety=interp(0.9,alphas,thetaAxis)
print '90% upper limit on theta: ',thetaNinety

#
# Plots
#
close('all')

figure()
plot(rateAxis,cbcRatePos)
axvline(rateNinety,color='r')

minorticks_on()
#grid(which='major',linestyle='-',alpha=0.5)
xlabel(r'$\textrm{Rate }[\textrm{Mpc}^{-3}\textrm{yr}^{-1}]$')
ylabel(r'$p({\mathcal R}_{\textrm{cbc}}|D,I)$')
subplots_adjust(bottom=0.17,left=0.17,top=0.95,right=0.98)
#grid(which='minor',linestyle='-',alpha=0.5)
savefig('rate_posterior_s6UL.eps')

figure()
plot(thetaAxis,thetaPos)
axvline(thetaNinety,color='r')
minorticks_on()
#grid(which='major',linestyle='--',alpha=0.5)
xlim(0,10)
#subplots_adjust(bottom=0.15,top=0.9,right=0.99)
xlabel(r'$\textrm{Jet Angle, }\theta~[\textrm{degrees}]$')
ylabel(r'$p(\theta|D,I)$')
subplots_adjust(bottom=0.15,left=0.125,top=0.95,right=0.98)
savefig('jet_angle_posterior_s6UL.eps')

show()


