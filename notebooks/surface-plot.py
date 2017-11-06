import pymc3 as pm
import numpy as np
import theano



from theano.compile.ops import as_op
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("/home/daniel/papers/thesis/thesis-style.mpl")

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


def beaming_violins(traces):

    width = 3.487 #* 2
    height = width / 1.618


    f, ax = plt.subplots(1,1, sharex=True, figsize = (width, height))
    #priors = ["U(0,1)", "Jeffreys", "$\delta(1)$", "$\delta(0.5)$"]
    print "|\t\t\t| Lower\t| MAP\t| Median\t| Upper\t|"
    print "|----------|"
    matplotlib.rcParams.update({'font.size': 10})
    pos = [.5, 1, 1.5, 2]


    o2_trace = traces
    #o2_trace = o2_traces[i]

        #i = i/2.0
    t_data = o2_trace[2000:]['angle'][np.isfinite(o2_trace[2000:]['angle'])]
    data = np.rad2deg(t_data)

    parts = ax.violinplot(data, [pos[0]], points=100, widths=0.3, vert= False,
                     #showmeans = True, showmedians=True, 
                      showmeans=False, showextrema=False, showmedians=False)

    lower_p, medians, upper_p = np.percentile(data, [2.5, 50, 97.50])
    lower, upper = pm.stats.hpd(t_data, alpha=0.05, transform=np.rad2deg)
    hist = np.histogram(data, bins = 90)
    MAP = hist[1][np.argmax(hist[0])]



    ax.hlines(pos[0], lower, upper, color='#333333', linestyle='-', lw=2, alpha = 0.5)

    ax.scatter( [lower, upper], [pos[0]]*2, marker='|', color='k', s=15, zorder=3)
    ax.scatter( [MAP], pos[0], marker='D', color='k', s=15, zorder=3)
    ax.scatter( [medians], pos[0], marker='s', color='k', s=15, zorder=3)
    #ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    print "| {} \t| {:.2f}\t| {:.2f}\t| {:.2f}\t| {:.2f}\t|".format(priors[i], lower, MAP, medians, upper)

    axis = ax
    axis.set_yticks(pos)
    axis.set_yticklabels(priors)
    axis.set_xlim([0, 52])
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, axis='x', which='major', linewidth=0.5)
    ax.grid(b=True, axis='y', which='major', linewidth=0)
    #ax.grid(b=True, which='minor', linewidth=0.5)
    ax.set_xlabel(r"Beaming Angle [$\theta$]")
    ax.set_ylabel(r"Prior Distribution on efficiency")
    ax.tick_params(axis='y',which='both',left='off')
    f.subplots_adjust(0.20, 0.15, .98, .95, wspace=0.05)
    #f.savefig("O2a_beaming_posteriors_violin.pdf")
    #f.savefig("O2a_beaming_posteriors_violin.png", dpi=300)
    return f


def background_rate_f(b, T, n):
    """
    
    """
    out = 0
    #n = int(n)
    for i in range(n+1):
        out += ((b*T)**i * np.exp(- b*T)) / np.math.factorial(i)
    return out

def log_background_rate(b, T, n):
    return np.log(background_rate_f(b, T, n))

def signal_rate_part(s, n, b, T):
    top_a = T * ((s + b) * T)**n 
    top_b = np.exp(-(s + b)*T)
    p = (top_a * top_b) / np.math.factorial(n)
    return theano.tensor.switch(theano.tensor.le(s, 0), 0, p)

#@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dscalar])
def log_signal_rate(s,n,b,T):
    #if theano.tensor.lt(0, s): return np.array([[0.0]])
    p = -log_background_rate(b,T,n) + np.log(signal_rate_part(s,n,b,T))
    return p

def number_mweg(volume):
    """
    Calculates the number of MWEGs in a volume, given in units of Mpc^3
    """
    return volume * (0.0116) 

import theano.tensor as T
from pymc3 import DensityDist, Uniform, Normal
from pymc3 import Model
from pymc3 import distributions


def grb_model_sigma(number_events, background_rate, 
              observation_time, volume,  grb_rate,
             efficiency_prior = "uniform"):
    with Model() as model:
        signal_rate = pm.DensityDist('signal_rate', 
                                     logp=lambda value: log_signal_rate(value, number_events, background_rate, observation_time), testval=number_events+2,
                           )

        #volume = pm.Normal("volume", volume, sigma_volume)
        
        n_galaxy = number_mweg(volume)
    
        cbc_rate = pm.Deterministic('cbc_rate', signal_rate / n_galaxy)# * n_galaxy)
        
        grb_rate = (grb_rate /  number_mweg(1e9))
        
        # Allow the efficiency prior to be switched-out
        if efficiency_prior == "uniform":
            efficiency = pm.Uniform('efficiency', 0,1, testval=0.5)
        elif efficiency_prior == "jeffreys":
            efficiency = pm.Beta('efficiency', 0.5, 0.5, testval = 0.3)
        elif isinstance(efficiency_prior, float):
            efficiency = efficiency_prior
        
        def cosangle(cbc_rate, efficiency, grb_rate):
            return T.switch((grb_rate >= cbc_rate*efficiency), -np.Inf, 
                                 (1.0 - ((grb_rate/(cbc_rate*efficiency)))))
        
        costheta = pm.Deterministic('cos_angle', cosangle(cbc_rate, efficiency, grb_rate))

        angle = pm.Deterministic("angle", theano.tensor.arccos(costheta))
        
        return model

horizon_int = 200 #200
events_max = 20

# make a plot of the beaming angle as a function of observation volume against number of detections
# O1 Scenarios
scenarios = []
for events in range(events_max):
    for horizon in np.linspace(10, 1000, horizon_int):

        ##

        volume = np.pi * (4./3.) * (2.26*horizon)**3
        print "Horizon: {}\t Events: {}\t MWEG: {}\t volume: {}".format(horizon, events, number_mweg(volume), volume)

        number_events = events # There were no BNS detections in O1
        background_rate = 0.01 # We take the FAR to be 1/100 yr
        observation_time = 1.0  # Years
        grb_rate = 10.0
        #for prior in priors:
        scenarios.append( grb_model_sigma(number_events, background_rate, observation_time, volume, grb_rate, efficiency_prior='jeffreys'))

        ##

traces = []
angles975 = []
angles025 = []
angles500 = []
uppers = []
lowers = []
samples = 100000

priors = ["jeffreys"]

for model in scenarios:
    with model:
        step = pm.Metropolis()
        trace = pm.sample(samples, step, )
        #trace = pm.sample(samples, step, )
        traces.append(trace)
        t_data = trace['angle'][trace['angle']>0]
        t_data = t_data[~np.isnan(t_data)]
        t_data = t_data[np.isfinite(t_data)]
        t_data = t_data[2000:]

        print "Mean CBC: {}".format(1e9*np.nanmean(trace['cbc_rate']))
        #print "Mean GRB: {}".format(np.nanmean(
        print "NANs: {}".format(np.sum([np.isnan(t_data)]))
        
        try:
            lower, upper = pm.stats.hpd(t_data, alpha=0.05, transform=np.rad2deg)
        except ValueError:
            lower, upper = np.nan, np.nan
            print "There weren't enough samples."

        print "lower: {}, upper: {}".format(lower, upper)
        #angles975.append(np.nanpercentile(trace['angle'][10000:], 97.5))
        #angles025.append(np.nanpercentile(trace['angle'][10000:], 2.5))

        #f, ax = plt.subplots(1,1)

        #print trace['signal_rate']
        
        #t_data = trace[2000:]['angle'][np.isfinite(trace[2000:]['angle'])]
        #data = np.rad2deg(t_data)

        # parts = ax.violinplot(data, 0.0, points=100, widths=0.3, vert= False,
        #                       #showmeans = True, showmedians=True, 
        #                       showmeans=False, showextrema=False, showmedians=False)

        #lower_p, medians, upper_p = np.percentile(data, [2.5, 50, 97.50])
        #lower, upper = pm.stats.hpd(t_data, alpha=0.05, transform=np.rad2deg)

        #mweg_viols.savefig("test.png")
        
        angles500.append(np.nanpercentile(trace['angle'][10000:], 50))
        lowers.append(lower)
        uppers.append(upper)
        np.savetxt("upper.dat", uppers)
        np.savetxt("lower.dat", lowers)
        np.savetxt("500perc.dat", angles500)

np.savetxt("upper.dat", uppers)
np.savetxt("lower.dat", lowers)
np.savetxt("975perc", angles975)
np.savetxt("500perc", angles500)
np.savetxt("025perc", angles025)


import scipy.ndimage
from matplotlib import colors, ticker, cm
from astropy import constants as c
from astropy import units as u
#data = np.rad2deg(angles975).reshape(20, 200)

width = 3.487 #* 2
height = width / 1.618

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



ax.set_xlim([0, 100]);

vth = 4./3 * np.pi #* (c.c * 1*u.year).to(u.megaparsec)
for scenario in scenarios.iteritems():
    ax.vlines(scenario[1], 0, 19, color='k', alpha=0.5, lw=2)
    x_bounds = ax.get_xlim()
    ax.annotate(s=scenario[0], xy =(((scenario[1]-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), xycoords='axes fraction', verticalalignment='right', rotation = 0)

consf = ax.contourf(np.linspace(vth*(10/2.26)**3/1e6, vth*(1000/2.26)**3/1e6, data.shape[1]), np.linspace(0, 19, data.shape[0]), 
             data, 
             levels = np.linspace(0, 90, 10), alpha = 0.4, cmap = "magma_r", lw=0
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

cb.set_label('Beaming angle upper limit [$^{\circ}$]')
f.subplots_adjust(0.1, 0.15, 1, .95, wspace=0.05)
f.savefig("volume_v_nevents.pdf")

import scipy.ndimage
from matplotlib import colors, ticker, cm
from astropy import constants as c
from astropy import units as u
data = np.rad2deg(angles025).reshape(20, 200)

width = 3.487 #* 2
height = width / 1.618

matplotlib.rcParams.update({'font.size': 6})

scenarios = {
    #"2015 - 2016": 0.45,
    #"B": 1.3,
    #"C": 6.5,
    "D": 20,
    "E": 40
            }

#height, width= 20, 10
f, ax = plt.subplots(1,1, sharex=True, figsize = (width, height))



ax.set_xlim([0, 100]);

vth = 4./3 * np.pi #* (c.c * 1*u.year).to(u.megaparsec)
for scenario in scenarios.iteritems():
    ax.vlines(scenario[1], 0, 19, color='k', alpha=0.5, lw=2)
    x_bounds = ax.get_xlim()
    ax.annotate(s=scenario[0], xy =(((scenario[1]-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), xycoords='axes fraction', verticalalignment='right', rotation = 0)


cmap = plt.cm.get_cmap("magma")
cmap.set_over("white")
    
consf = ax.contourf(np.linspace(vth*(10/2.26)**3/1e6, vth*(1000/2.26)**3/1e6, data.shape[1]), np.linspace(0, 19, data.shape[0]), 
             data, 
             levels = np.linspace(0,10, 11), alpha = 0.4, cmap = cmap, lw=0, extend="max"
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
cb = f.colorbar(consf)
cb.set_label('Beaming angle lower limit [$^{\circ}$]')

#cb3.set_label('Custom extension lengths, some other units')

ax.set_xlabel(r"Search 4-volume, $VT\, [\times10^6\, \rm{Mpc}^3 \rm{yr}]$")
ax.set_ylabel(r"GW BNS event rate [${\rm yr}^{-1}$]")
f.subplots_adjust(0.1, 0.15, 1, .95, wspace=0.05)
f.savefig("volume_v_nevents_lower.pdf")
