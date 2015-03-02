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
from scipy.stats import beta as beta_dist
from scipy.optimize import curve_fit
from optparse import OptionParser

#   fig_width_pt = 245.26653  # Get this from LaTeX using \showthe\columnwidth
#   #fig_width_pt=600.
#   inches_per_pt = 1.0/72.27               # Convert pt to inch
#   golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
#   fig_width = fig_width_pt*inches_per_pt  # width in inches
#   fig_height = fig_width*golden_mean      # height in inches
#   fig_size =  [fig_width,fig_height]
#   rcParams.update(
#           {'axes.labelsize': 10,
#           'text.fontsize':   10,
#           'legend.fontsize': 10,
#           'xtick.labelsize': 10,
#           'ytick.labelsize': 10,
#           'text.usetex': True,
#           'figure.figsize': fig_size,
#           'font.family': "serif",
#           'font.serif': ["Times"],
#           'savefig.dpi': 200,
#           'xtick.major.size':8,
#           'xtick.minor.size':4,
#           'ytick.major.size':8,
#           'ytick.minor.size':4
#           })  

def parser():
    """
    Read the input from the command line.
    """
    parser = OptionParser()

    # Naming scheme
    parser.add_option('--outputname', default="TEST", type='string',
                      help="Name to append to output files [Default=None]")

    # Priors
    parser.add_option('--delta-eff-prior', default=False, action='store_true',
            help="GRB efficiency prior is a delta function [requires --eff-val]")
    parser.add_option('--eff-val', default=1., type='float',
            help="GRB efficiency (for delta function prior)")

    parser.add_option('--flat-eff-prior', default=False, action='store_true',
            help="GRB efficiency prior is a (linear) uniform distribution")
    parser.add_option('--flat-eff-bounds', default='0.001,1',
            help="lower,upper bounds for uniform efficiency prior")

    parser.add_option('--log-eff-prior', default=False, action='store_true',
            help="GRB efficiency prior is a (logarithmic) uniform distribution")
    parser.add_option('--log-eff-bounds', default='0.001,1',
            help="lower,upper bounds for uniform efficiency prior")

    parser.add_option('--berno-eff-prior', default=False, action='store_true',
            help="GRB efficiency prior is the Bernoulli Jeffrey's prior (beta"
            " with params 1/2, 1/2)")

    parser.add_option('--beta-eff-prior', default=False, action='store_true',
            help="GRB efficiency prior is a beta function")
    parser.add_option('--beta-vals', default='2,5',
            help="Shape parameters for beta distribution")

    (opts, args) = parser.parse_args()

    # sanity checks
    if opts.delta_eff_prior + opts.flat_eff_prior \
            + opts.berno_eff_prior + opts.beta_eff_prior \
            + opts.log_eff_prior > 1:
                print >> sys.stderr, "ERROR only one type of prior allowed\n"
                parser.print_help()
                sys.exit(-1)


    return(opts,args)

def cbcRatePosteriorNull(eps,rateAxis):
    """
    Returns the value of the posterior on the cbc Rate for Lambda=0
    """
    return eps*exp(-rateAxis*eps)

def rateFromTheta(theta,grb_efficiency,grbRate):
    """
    Returns Rcbc = Rgrb / (1-cos(theta))
    """
    return grbRate / ( grb_efficiency*(1.-cos(theta * pi / 180)) )

def compute_jacobian(grbRate,grb_efficiency,theta):
    """
    Compute the Jacboian for the transformation from rate to angle
    """
    denom=grb_efficiency*(cos(theta * pi/180)-1)
    return abs(2*grbRate * sin(theta * pi / 180) / denom**2)

def compute_beta(theta):
    """
    Compute v/c, assuming theta ~ 1/Lorentz factor
    """
    gamma = 1.0/(theta * pi/180)
    beta = sqrt(1 - 1./gamma**2)
    return beta,gamma


############################
# MAIN
#

# Prior type can be a delta on 
opts,args=parser()

#
# --- Construct Efficiency Prior ---
#
if opts.delta_eff_prior:
    # delta function prior (known efficiency)
    grb_efficiency_axis=array([opts.eff_val])
    grb_efficiency_prior=array([1.])

    outputname=opts.outputname+"_deltaEffPrior-"+str(opts.eff_val)

elif opts.flat_eff_prior:
    # linear uniform prior
    lower_bound,upper_bound=opts.flat_eff_bounds.split(',')
    grb_efficiency_axis=linspace(float(lower_bound),float(upper_bound),1000)
    grb_efficiency_prior=1./(float(upper_bound)-float(lower_bound)) * \
            ones(len(grb_efficiency_axis))

    outputname=opts.outputname+"_flatEffPrior-%s-%s"%(lower_bound,upper_bound)

elif opts.log_eff_prior:
    # logarithmic uniform prior
    lower_bound,upper_bound=opts.log_eff_bounds.split(',')
    grb_efficiency_axis=linspace(float(lower_bound),float(upper_bound),1000)

    Norm=log(float(upper_bound)/float(lower_bound))
    grb_efficiency_prior = Norm  / grb_efficiency_axis 

    outputname=opts.outputname+"_logEffPrior-%s-%s"%(lower_bound,upper_bound)

elif opts.berno_eff_prior:
    # bernoulli trial parameter
    prior_dist=beta_dist(0.5,0.5)
    grb_efficiency_axis=linspace(0.01,0.99,1000)
    grb_efficiency_prior=prior_dist.pdf(grb_efficiency_axis)

    outputname=opts.outputname+"_bernoEffPrior"

elif opts.beta_eff_prior:
    # beta distribution prior
    beta_vals=opts.beta_vals.split(',')
    prior_dist=beta_dist(float(beta_vals[0]),float(beta_vals[1]))

    grb_efficiency_axis=linspace(0.01,0.99,1000)
    grb_efficiency_prior=prior_dist.pdf(grb_efficiency_axis)

    outputname=opts.outputname+"_betaEffPrior-%s-%s"%(beta_vals[0],beta_vals[1])


#
# --- Construct Measured Rate Posterior ---
#

# Here's the simple analytic expression:
eps=-1*log(1-0.9)/1.3e-4

rateAxis=linspace(1e-8,5e-4,5000)
cbcRatePos=cbcRatePosteriorNull(eps,rateAxis)
alphas=[]
for bin in rateAxis:
    alphas.append(trapz(y=cbcRatePos[rateAxis<bin], x=rateAxis[rateAxis<bin], dx=diff(rateAxis)[0]))
rateNinety=interp(0.9,alphas,rateAxis)
print '%d %0.3e'%(eps,rateNinety)

close('all')

figure()
plot(rateAxis,cbcRatePos, color='k',label='S6 result')
axvline(rateNinety,color='k', linestyle='--',label=r'$90\%$ U.L.')

minorticks_on()
#grid(which='major',linestyle='-',alpha=0.5)
#xlabel(r'$\textrm{Rate }[\textrm{Mpc}^{-3}\textrm{yr}^{-1}]$')
xlabel(r'Binary Coalescence Rate, $R$ [Mpc$^{-3}$yr$^{-1}$]')
ylabel(r'$p(R|D,I)$')
#subplots_adjust(bottom=0.2,left=0.2,top=0.95,right=0.98)
legend()
tight_layout()
#grid(which='minor',linestyle='-',alpha=0.5)
savefig('rate_posterior_s6UL.eps')
savefig('rate_posterior_s6UL.png')
savefig('rate_posterior_s6UL.pdf')

#
# --- Compute Angle Posterior ---
#

# Transform to jet angle:
thetaAxis=arange(0.01,90,0.1)
grbRate=10./ 1e9  # 10 is Gpc^-3...
thetaPos=zeros(len(thetaAxis))

# Loop through jet angles:
# 1) compute jacobian for transformation
# 2) find the Rcbc corresponding to this Rgrb and theta (i.e., get p(theta) in
# terms of Rcbc).  Then multiply by compute_jacobian to get correct density
# 3) interpolate the Rcbc posterior to this Rcbc

for t,theta in enumerate(thetaAxis):
    print "evaluating angle %d of %d"%(t,len(thetaAxis))

    if len(grb_efficiency_axis)>1:
        # do the efficiency marginalisation
        integrand=zeros(len(grb_efficiency_axis))
        for e,eff in enumerate(grb_efficiency_axis):
                jacobian=compute_jacobian(grbRate,eff,theta)    
                cbcRate=rateFromTheta(theta,eff,grbRate)

                cbcRatePosVal=cbcRatePosteriorNull(eps,cbcRate)

                integrand[e]=grb_efficiency_prior[e]*cbcRatePosVal*jacobian

        thetaPos[t]=trapz(integrand,grb_efficiency_axis)
    else:
        # no marginalisation needed
        jacobian=compute_jacobian(grbRate,grb_efficiency_axis[0],theta)    
        cbcRate=rateFromTheta(theta,grb_efficiency_axis[0],grbRate)
        cbcRatePosVal=cbcRatePosteriorNull(eps,cbcRate)
        thetaPos[t]=cbcRatePosVal*jacobian


# Ensure normalisation to unity:
N=trapz(thetaPos,thetaAxis)
thetaPos/=trapz(thetaPos,thetaAxis)

# Now integrate to get UL
alphas=[]
for theta in thetaAxis:
    alphas.append(trapz(thetaPos[thetaAxis<theta],thetaAxis[thetaAxis<theta]))
thetaNinety=interp(0.9,alphas,thetaAxis)
print '90% upper limit on theta: ',thetaNinety

#
# Dump results to file
#
savez(outputname, rateAxis=rateAxis, cbcRatePos=cbcRatePos,
        thetaAxis=thetaAxis, thetaPos=thetaPos, thetaNinety=thetaNinety,
        grb_efficiency_axis=grb_efficiency_axis,
        grb_efficiency_prior=grb_efficiency_prior)

#
# Plots
#

figure()
plot(thetaAxis,thetaPos,'-', color='k')
axvline(thetaNinety, color='k', linestyle='--',label=r'$90\%$ U.L')
legend()
minorticks_on()
#grid(which='major',linestyle='--',alpha=0.5)
#xlim(0,10)
#subplots_adjust(bottom=0.15,top=0.9,right=0.99)
#xscale('log')
xlabel(r'Jet Angle, $\theta$ [deg]')
ylabel(r'$p(\theta|D,I)$')
subplots_adjust(bottom=0.2,left=0.2,top=0.95,right=0.98)
savefig('jet_angle_posterior_s6UL_%s.eps'%outputname)
savefig('jet_angle_posterior_s6UL_%s.png'%outputname)
savefig('jet_angle_posterior_s6UL_%s.pdf'%outputname)

#show()


