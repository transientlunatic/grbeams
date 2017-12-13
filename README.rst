==============================================================================================
Constraints On Short, Hard Gamma-Ray Burst Beaming Angles From Gravitational Wave Observations
==============================================================================================

.. image:: https://zenodo.org/badge/54910948.svg
   :target: https://zenodo.org/badge/latestdoi/54910948

	    

This is the repository for the analysis and the paper related to the method for inferring the GRB opening angle from the rates of gamma ray bursts and neutron star merger events.

The pre-print of the paper describing this work and its results is available on the `arXiv`_ as document 1712.02585.

This document also has LIGO-DCC document ID `LIGO-P1600102`_.


Abstract
--------

The first detection of a binary neutron star merger, GW170817, and an
associated short gamma-ray burst confirmed that neutron star mergers
are responsible for at least some of these bursts. The prompt gamma
ray emission from these events is thought to be highly
relativistically beamed. We present a method for inferring limits on
the extent of this beaming by comparing the number of short gamma-ray
bursts observed electromagnetically to the number of neutron star
binary mergers detected in gravitational waves. We demonstrate that an
observing run comparable to the expected Advanced LIGO 2016--2017 run
would be capable of placing limits on the beaming angle of
approximately θ ∈ (2.88°,14.15°), given one binary neutron star
detection. We anticipate that after a year of observations with
Advanced LIGO at design sensitivity in 2020 these constraints would
improve to θ ∈ (8.10°,14.95°).


Citing this work
----------------

You can cite our arXiv preprint (BibTeX):

::
   
   @ARTICLE{2017arXiv171202585W,
      author = {{Williams}, D. and {Clark}, J.~A. and {Williamson}, A.~R. and 
	{Heng}, I.~S.},
       title = "{Constraints On Short, Hard Gamma-Ray Burst Beaming Angles From Gravitational Wave Observations}",
       journal = {ArXiv e-prints},
       archivePrefix = "arXiv",
       eprint = {1712.02585},
       primaryClass = "astro-ph.HE",
       keywords = {Astrophysics - High Energy Astrophysical Phenomena, General Relativity and Quantum Cosmology},
       year = 2017,
       month = dec,
       adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171202585W},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

and you can also cite our data release:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1066019.svg
   :target: https://doi.org/10.5281/zenodo.1066019

::

   @article{williams, daniel_clark, james_williamson, andrew_heng, ik
   siong_2017, title={Constraints On Short, Hard Gamma-Ray Burst
   Beaming Angles From Gravitational Wave Observations : Supplementary
   Material}, DOI={10.5281/zenodo.1066019}, abstractNote={<p>Data and
   code to produce the analysis presented in &quot;Constraints On
   Short, Hard Gamma-Ray Burst Beaming Angles From<br> Gravitational
   Wave Observations&quot;.</p>}, note={This is the version of the
   data release which corresponds to v1 of the paper on arXiv. DW is
   supported by the Science and Technology Research Council (STFC)
   grant ST/N504075/1. JC acknowledges support from NSF awards
   PHYS-1505824 and PHYS-1505524SH. ISH is supported by STFC grant
   ST/L000946/1. ARW acknowledges support from the Netherlands
   Organization for Scientific Research through the NWO TOP Grant
   No. 62002444­­-Nissanke. This document has been assigned the
   control number LIGO-P1600102 by the LIGO document control centre.},
   publisher={Zenodo}, author={Williams, Daniel and Clark, James and
   Williamson, Andrew and Heng, Ik Siong}, year={2017}, month={Dec}}

   
   

Reproducing our results
-----------------------

This repository contains all of the analysis code required to
reproduce the results presented in the paper, however the surface
plots require a fairly large amount of time to produce, so for
convenience we have provided the data for these in the `/data`
directory. The majority of the analysis is conducted in `jupyter`
format notebooks, and these can be found in the `notebooks` directory,
which has its own `README` file. To reproduce the analysis for the
surface plots see the `surface-plot.py` script in the `scripts`
directory.

You can install the requirements for this analysis in a virtual
environment by first running

::
   
   pip install -r requirements.txt

inside this directory.

.. _here: https://git.ligo.org/daniel-williams/grb-beaming/-/jobs/7589/artifacts/file/final_paper/grb_beams_paper.pdf
.. _LIGO-P1600102: https://dcc.ligo.org/LIGO-P1600102
.. _arXiv: https://arxiv.org/abs/1712.02585
