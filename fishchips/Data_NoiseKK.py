import numpy as np
import matplotlib.pyplot as plt
ell = np.array([42.0, 122.0, 202.0, 282.0, 362.0, 442.0, 522.0, 602.0, 682.0, 762.0 ,842.0 ,922.0 ,1002.0 ,1082.0 ,1162.0 ,1242.0 ,1322.0 ,1402.0 ,1482.0 ,1562.0 ,1642.0 ,1722.0 ,1802.0 ,1882.0 ,1962.0 ,2042.0 ,2122.0 ,2202.0 ,2282.0 ,2362.0 ,2442.0 ,2522.0 ,2602.0 ,2682.0 , 2762.0, 2842.0, 2922.0]) 
nlkk = np.array([1.02886566516913e-08,1.1194860550459957e-08,1.2505619136465732e-08,1.3639000292859354e-08,1.4726157985091727e-08,1.5725847204661933e-08,1.6938661270395064e-08,1.8682705809355073e-08,2.112299958392952e-08,2.3537204158565772e-08,2.6496287215049565e-08,3.025903245034381e-08,3.4250486637297883e-08,3.858609285601067e-08,4.3227517165236133e-08,4.881779318529917e-08,5.4637529106849264e-08,6.091794461679896e-08,6.748938330785925e-08,7.478835883220224e-08,8.228241718672807e-08,8.893682840859781e-08,9.60522674909952e-08,1.0384832218496256e-07,1.1192068433016309e-07,1.199852307972035e-07,1.3053890509577023e-07,1.43859054270254e-07,1.5863511614121002e-07,1.7420892487626716e-07,1.9100543065917838e-07,2.102343953615504e-07,2.2934157407025104e-07,2.4737779899712124e-07,2.6884587664392953e-07,2.918082225405003e-07,3.080468562881667e-07])

def interpolate_noiseKK(l_min, l_max, noise_K = [], ell_K = []):
    
    ellK = np.linspace(l_min, l_max, (l_max-l_min+1))
    if noise_K = []:
        nlK = np.interp(ellK, ell, nlkk)
    else: 
        nlK = np.interp(ellK, ell_K, noise_K)
    
    noise_K = np.zeros(l_max+1, 'float64')
    for l in range(l_min, l_max+1):
        noise_K[l] = nlK[l-l_min]
    return noise_K