#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:28:46 2018

@author: aizhana
"""

import fishchips.experiments as experiments
from fishchips.cosmo import Observables
import fishchips.util

from classy import Class
import numpy as np
import matplotlib.pyplot as plt

pars = np.array( ['omega_b', 'omega_cdm', 'h',  'A_s', 'n_s', 'tau_reio'])
centers = np.array([0.02230,  0.1188,  0.6774,  2.142e-9, 0.9619, 0.66])
steps = np.array([0.0002230,  0.001188,  0.006774,  2.142e-11, 0.009619, 0.0066])


obs = Observables(parameters=pars,
                  fiducial=centers,
                  left=centers-steps,
                  right=centers+steps)
# generate a template CLASS python wrapper configuration
classy_template = {'output': 'tCl pCl lCl',
                   'l_max_scalars': 2500,
                   'lensing': 'yes'}
# add in the fiducial values too
classy_template.update(dict(zip(obs.parameters, obs.fiducial)))
# generate the fiducial cosmology
obs.compute_cosmo(key='CLASS_fiducial', classy_dict=classy_template)
# generate an observables dictionary, looping over parameters
for par, par_left, par_right in zip(obs.parameters, obs.left, obs.right):
    classy_left = classy_template.copy()
    classy_left[par] = par_left
    classy_right = classy_template.copy()
    classy_right[par] = par_right
    # pass the dictionaries full of configurations to get computed
    obs.compute_cosmo(key=par + '_CLASS_left', classy_dict=classy_left)
    print(par + "_CLASS_left")
    obs.compute_cosmo(key=par + '_CLASS_right', classy_dict=classy_right)
    print(par + "_CLASS_right")
    
print(obs.cosmos['CLASS_fiducial'].T_cmb())


example_Planck = experiments.get_PlanckPol_combine()
fisher = example_Planck[0].get_fisher(obs)+example_Planck[1].get_fisher(obs)+example_Planck[2].get_fisher(obs)
cov = np.linalg.inv(fisher)
fishchips.util.plot_triangle(obs, cov);