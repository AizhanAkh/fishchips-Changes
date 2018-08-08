#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:06:48 2018

@author: aizhana
"""



import numpy as np
import matplotlib.pyplot as plt
import fishchips.experiments_highell as experiments
import fishchips.util
from fishchips.plots import plot_ell

# Get all data points for all parameters
# from output .dat files

# after "=" write path to the .dat file with Pk data (..._pk) or Cl data (..._cl)
# Output files (from .ini files above) I used are in PT-check/class/test_output/
# for example, lcdm_pk = "/Users/aizhan.akh/Documents/Projects/PT-check/class/test_output/lcdm_pk.dat"

# for LCDM
fiducial_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/lcdm_pt_highl_cl_lensed.dat")[:,:]

# for dmeff
dmeff_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_dmeff_highl_cl_lensed.dat")[:,:]

#for omega_b              
omega_b_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_omega_b_highl_cl_lensed.dat")[:,:]
omega_b_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_omega_b_highl_cl_lensed.dat")[:,:]

#for omega_cdm             
omega_cdm_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_omega_cdm_highl_cl_lensed.dat")[:,:] 
omega_cdm_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_omega_cdm_highl_cl_lensed.dat")[:,:] 

#for tau              
tau_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_tau_highl_cl_lensed.dat")[:,:] 
tau_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_tau_highl_cl_lensed.dat")[:,:]

#for h             
h_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_h_highl_cl_lensed.dat")[:,:] 
h_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_h_highl_cl_lensed.dat")[:,:]

#for As              
As_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_As_highl_cl_lensed.dat")[:,:] 
As_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_As_highl_cl_lensed.dat")[:,:]

#for ns              
ns_l_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_l_ns_highl_cl_lensed.dat")[:,:] 
ns_r_data = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/lcdm_pt_r_ns_highl_cl_lensed.dat")[:,:]

# Get all data points for all parameters
# from output .dat files

# after "=" write path to the .dat file with Pk data (..._pk) or Cl data (..._cl)
# Output files (from .ini files above) I used are in PT-check/class/test_output/
# for example, lcdm_pk = "/Users/aizhan.akh/Documents/Projects/PT-check/class/test_output/lcdm_pk.dat"

# for LCDM
fiducial_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/linear/lcdm_pt_highl_linear_cl_lensed.dat")[:,:]

# for dmeff
dmeff_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_dmeff_highl_linear_cl_lensed.dat")[:,:]

#for omega_b              
omega_b_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_omega_b_highl_linear_cl_lensed.dat")[:,:]
omega_b_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_omega_b_highl_linear_cl_lensed.dat")[:,:]

#for omega_cdm             
omega_cdm_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_omega_cdm_highl_linear_cl_lensed.dat")[:,:] 
omega_cdm_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_omega_cdm_highl_linear_cl_lensed.dat")[:,:] 

#for tau              
tau_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_tau_highl_linear_cl_lensed.dat")[:,:] 
tau_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_tau_highl_linear_cl_lensed.dat")[:,:]

#for h             
h_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_h_highl_linear_cl_lensed.dat")[:,:] 
h_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_h_highl_linear_cl_lensed.dat")[:,:]

#for As              
As_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_As_highl_linear_cl_lensed.dat")[:,:] 
As_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_As_highl_linear_cl_lensed.dat")[:,:]

#for ns              
ns_l_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_l_ns_highl_linear_cl_lensed.dat")[:,:] 
ns_r_data_linear = np.genfromtxt("/u/aizhana/Projects/CodeCombined/Fisher/output/highl/linear/lcdm_pt_r_ns_highl_linear_cl_lensed.dat")[:,:]

pars = np.array( ['omega_b', 'omega_cdm', 'h',  'A_s', 'n_s', 'tau_reio', 'sigma_dmeff'])
centers = np.array([0.02230,  0.1188,  0.6774,  2.142e-9, 0.9619, 0.066, 0.0])
steps = np.array([0.0002230,  0.001188,  0.006774,  2.142e-11, 0.009619, 0.00066, 2.1e-26])

fiducial = {}
dmeff = {}       
omega_b_l = {}
omega_b_r = {} 
omega_cdm_l = {} 
omega_cdm_r = {}             
tau_l = {}
tau_r = {}           
h_l = {}
h_r = {}            
As_l = {}
As_r = {} 
ns_l = {}
ns_r = {}

channels = ['ell', 'tt', 'ee', 'te', 'bb', 'pp', 'tp', 'ep']
for i in channels:
    j = channels.index(i)
    fiducial[i] = fiducial_data[:, j]
    dmeff[i] = dmeff_data[:, j]       
        
    omega_b_l[i] = omega_b_l_data[:, j] 
    omega_b_r[i] = omega_b_r_data[:, j] 
        
    omega_cdm_l[i] = omega_cdm_l_data[:, j]  
    omega_cdm_r[i] = omega_cdm_r_data[:, j]              
        
    tau_l[i] = tau_l_data[:, j]
    tau_r[i] = tau_r_data[:, j]             
        
    h_l[i] = h_l_data[:, j] 
    h_r[i] = h_r_data[:, j]             
        
    As_l[i] = As_l_data[:, j] 
    As_r[i] = As_r_data[:, j]
        
    ns_l[i] = ns_l_data[:, j] 
    ns_r[i] = ns_r_data[:, j]
        
cl_left = []
cl_right = []

cl_left.append(omega_b_l)
cl_left.append(omega_cdm_l)
cl_left.append(h_l)
cl_left.append(As_l)
cl_left.append(ns_l)
cl_left.append(tau_l)
cl_left.append(fiducial)

cl_right.append(omega_b_r)
cl_right.append(omega_cdm_r)
cl_right.append(h_r)
cl_right.append(As_r)
cl_right.append(ns_r)
cl_right.append(tau_r)
cl_right.append(dmeff)

#cl_right[1]['tt']

example_Planck = experiments.get_PlanckPol_combine()
fisher = example_Planck[0].get_fisher_changed(fiducial, pars, cl_right, cl_left, steps)+example_Planck[1].get_fisher_changed(fiducial, pars, cl_right, cl_left, steps)+example_Planck[2].get_fisher_changed(fiducial, pars, cl_right, cl_left, steps)
cov = np.linalg.inv(fisher)

import fishchips.corner as corner

def unitize_cov(imp_cov, scales):
    imp_cov = imp_cov.copy()
    npar = imp_cov.shape[0]
    for i in range(npar):
        for j in range(npar):
            imp_cov[i,j] *= scales[i] * scales[j]
    return imp_cov

def get_samps(inp_cov, inp_means, num=int(1e8)):
    """
    Generate samples from a covariance matrix and input means.
    
    Parameters
    ----------
        inp_cov (2D numpy array) : covariance matrix from Fisher
        inp_means (1D numpy array) : mean values (mu), fiducial from Fisher
        
    Returns
    -------
        2D numpy array with each row corresponding to one random draw 
        from the multivariate Gaussian
    """
    samps = np.random.multivariate_normal( np.array(inp_means)/np.sqrt(np.diag(inp_cov)), 
                                           unitize_cov(inp_cov,1./np.sqrt(np.diag(inp_cov))), int(1e7))
    samps = samps[samps.T[-1]>0]
    for i in range(inp_cov.shape[0]):
        samps.T[i] *= np.sqrt(inp_cov[i,i])
        
    return samps
        
def convert_to_cc_array(sig, mass_X):
    """utility function for converting sigma_p to the coupling constant
    
    Parameters
    ----------
        sig (float) : the cross section, sigma_p, possibly a numpy 
        mass_X (float) : dark matter particle mass in GeV
        
    Returns
    -------
        Numpy array that 
    """
    mp = 0.9382720813
    GeV = 1./(0.19732705e-15)
    cc_conversion = (1./( (246.22 * GeV)**4 * np.pi)) * \
        ( (mass_X * mp) / (mass_X + mp) * GeV )**2
    #cc_conversion = (1./( (246.22 * GeV)**4 * np.pi)) * \
        #( (mass_X * mp) / (mass_X + mp) * GeV )**2
    derived_cc = np.sqrt( sig * 1.e-4 / cc_conversion )
    return derived_cc


def get_95_exclusion(input_cov):
    """convenience function for turning a covariance matrix into a 95% exclusion.
    
    This function is specifically for the sigma_p case with a flat prior on the
    coupling, and positive definite cross section.
    
    Parameters
    ----------
        input_cov (numpy array) : covariance matrix
        
    Returns
    -------
        float, 95% upper limit for a fidcuial centered on zero
    """
    # NOTE: sigma_p MUST BE THE LAST VARIABLE
    samps = get_samps(input_cov, forecast_means_list, num=int(1e8))
    samps = samps[samps.T[-1]>0]
    onesig, twosig = corner.quantile(samps[:,-1], 
                                 [0.68,0.95], 
                                 weights=1./np.sqrt(samps.T[-1]))
    return twosig
    
fiducial_linear = {}
dmeff_linear = {}       
omega_b_l_linear = {}
omega_b_r_linear = {} 
omega_cdm_l_linear = {} 
omega_cdm_r_linear = {}             
tau_l_linear = {}
tau_r_linear = {}           
h_l_linear = {}
h_r_linear = {}            
As_l_linear = {}
As_r_linear = {} 
ns_l_linear = {}
ns_r_linear = {}

channels = ['ell', 'tt', 'ee', 'te', 'bb', 'pp', 'tp', 'ep']
for i in channels:
    j = channels.index(i)
    fiducial_linear[i] = fiducial_data_linear[:, j]
    dmeff_linear[i] = dmeff_data_linear[:, j]       
        
    omega_b_l_linear[i] = omega_b_l_data_linear[:, j] 
    omega_b_r_linear[i] = omega_b_r_data_linear[:, j] 
        
    omega_cdm_l_linear[i] = omega_cdm_l_data_linear[:, j]  
    omega_cdm_r_linear[i] = omega_cdm_r_data_linear[:, j]              
        
    tau_l_linear[i] = tau_l_data_linear[:, j]
    tau_r_linear[i] = tau_r_data_linear[:, j]             
        
    h_l_linear[i] = h_l_data_linear[:, j] 
    h_r_linear[i] = h_r_data_linear[:, j]             
        
    As_l_linear[i] = As_l_data_linear[:, j] 
    As_r_linear[i] = As_r_data_linear[:, j]
        
    ns_l_linear[i] = ns_l_data_linear[:, j] 
    ns_r_linear[i] = ns_r_data_linear[:, j]
        
cl_left_linear = []
cl_right_linear = []

cl_left_linear.append(omega_b_l)
cl_left_linear.append(omega_cdm_l)
cl_left_linear.append(h_l)
cl_left_linear.append(As_l)
cl_left_linear.append(ns_l)
cl_left_linear.append(tau_l)
cl_left_linear.append(fiducial)

cl_right_linear.append(omega_b_r)
cl_right_linear.append(omega_cdm_r)
cl_right_linear.append(h_r)
cl_right_linear.append(As_r)
cl_right_linear.append(ns_r)
cl_right_linear.append(tau_r)
cl_right_linear.append(dmeff)
