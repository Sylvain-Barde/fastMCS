# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:37:29 2024

This files provides a demo for the fast implementation of the Model Confidence 
Set algorithm (fastMCS). In particular, it demonstrates the ability to update
an existing MCS estimation with extra model losses.

@author: Sylvain Barde, University of Kent
"""
import numpy as np

from matplotlib import pyplot as plt
from numpy.random import default_rng
from scipy.stats import norm
from fastMCS import mcs


def genLosses(lmbda, rho, phi, M, N, seed = 0, shuffle = True):
    """
    Generate synthetic losses for the benchmarking exercise. Uses the 
    simulation experimet outlined in section 5.1 (page 477) of:
    
    Hansen, P.R., Lunde, A. and Nason, J.M., 2011. The model confidence set. 
    Econometrica, 79(2), pp.453-497.

    Parameters
    ----------
    lmbda : float
        Lambda parameter, controls the relative performance of different models
        (higher lambda -> models are easier to distinguish)
    rho : float
        Controls the correlation between model losses.
    phi : float
        Controls the level of conditional heteroskedasticity (GARCH effects) in
        model losses
    M : Int
        Number of models desired
    N : int
        Number of observations
    seed : flaot, optional
        Seed for the random number generator. The default is 0.
    shuffle : boolean, optional
        Controls whether losses are returned ordered or shuffled.
        - If set to False, losses are returned 'as generated', ordered with 
          model 1 displayignt ht lowest average loss and model M the highest.
        - If set to True, columns of L are shuffled so that the average 
          performance of models is randomised
        The default is True.

    Returns
    -------
    L : ndarray
        2D ndarray of synthetic losses. Structure is N x M

    """
    
    rng = default_rng(seed = seed)
    
    theta = lmbda*np.arange(0,M)/((M-1)*N**0.5)

    S = np.ones([M,M])*rho + np.diag(np.ones(M) - rho)
    Sroot = np.linalg.cholesky(S)
    X = Sroot @ norm.ppf(rng.random((M,N)))   # Correlated shocks
    
    e = norm.ppf(rng.random(N+1))
    y = np.zeros(N+1)
    y[0] = -phi/(2*(1+phi)) + e[0]*phi**0.5
    for i in range(1,N+1):
        y[i] = -phi/(2*(1+phi)) + y[i-1]*phi + e[i]*phi**0.5
    
    a = np.exp(y[1:N+1])
    
    L = a[:,None]*X.transpose()
    L /= np.std(L,axis = 0)[None,:]
    L += theta[None,:]
    if shuffle:
        # generate column permutation
        perm = np.arange(0,M)
        rng.shuffle(perm)
        L = L[:,perm]
    
    return L

# Set some parameters and generate losses
lmbda, rho, phi = (22,0.5,0.4)
numMods = 500                   # pick an even number!
numObs = 250
seed = 50
B = 1000
b = 10

# Generate synthetic losses for the comparison exercise
losses = genLosses(lmbda, rho, phi, numMods, numObs, seed = seed)

#------------------------------------------------------------------------------
# Run the original elimination MCS
mcsEst0 = mcs(seed = seed)
mcsEst0.addLosses(losses)
mcsEst0.run(B,b, bootstrap = 'stationary', algorithm = 'elimination')

# Get 90% MCS  
incl0, excl0 = mcsEst0.getMCS(alpha = 0.1)
rank0 = np.concatenate((excl0,incl0))
tScore0 = mcsEst0.tScore
pVals0 = mcsEst0.pVals

# Get run time and memory, print output
t0 = mcsEst0.stats[0]['time']       
mem0 = mcsEst0.stats[0]['memory use']*10**-6
print('\n Computational requirements:')
print(' Time (sec): {:.3f}    Memory (MB): {:.3f}'.format(t0,mem0))

#------------------------------------------------------------------------------
# Run the 2-pass fast MCS
mcsEst2 = mcs(seed = seed)
mcsEst2.addLosses(losses)
mcsEst2.run(B,b, bootstrap = 'stationary', algorithm = '2-pass')

# Get 90% MCS  
incl2, excl2 = mcsEst2.getMCS(alpha = 0.1)
rank2 = np.concatenate((excl2,incl2))
tScore2 = mcsEst2.tScore
pVals2 = mcsEst2.pVals

# Get run time and memory, compare model rankings and P-values
t2 = mcsEst2.stats[0]['time']       
mem2 = mcsEst2.stats[0]['memory use']*10**-6
print('\n Computational requirements:')
print(' Time (sec): {:.3f}    Memory (MB): {:.3f}'.format(t2,mem2))

print('\n Model rankings the same as elimination? - {}'.format(
        np.array_equal(rank2,rank0))
      )
print(' Model p-values the same as elimination? - {}'.format(
        np.array_equal(pVals2,pVals0))
     )
print(' Mean absolute deviation in P-values from elimination: {:.3f}'.format(
        np.mean(np.abs(pVals2-pVals0)))
      )
#------------------------------------------------------------------------------
# Run an updating demonstration
losses_1 = losses[:,0:int(numMods/2)]
losses_2 = losses[:,int(numMods/2):numMods]

# Run the 2-pass fast MCS on the first part of the losses
mcsEst1 = mcs(seed = seed)
mcsEst1.addLosses(losses_1)
mcsEst1.run(B,b, bootstrap = 'stationary', algorithm = '2-pass')

# Get run time and memory
t1 = mcsEst1.stats[0]['time']       
mem1 = mcsEst1.stats[0]['memory use']*10**-6
print('\n Computational requirements(losses_1):')
print(' Time (sec): {:.3f}    Memory (MB): {:.3f}'.format(t1,mem1))

# Update the existing MCS with the second part of losses
mcsEst1.addLosses(losses_2)
mcsEst1.run(B,b)            # This will run with the 1-pass method

# Get 90% MCS  
incl1, excl1 = mcsEst1.getMCS(alpha = 0.1)
rank1 = np.concatenate((excl1,incl1))
tScore1 = mcsEst1.tScore
pVals1 = mcsEst1.pVals

# Get run time and memory, compare model rankings and P-values
t1 = mcsEst1.stats[-1]['time']       
mem1 = mcsEst1.stats[-1]['memory use']*10**-6
print('\n Computational requirements (losses_2):')
print(' Time (sec): {:.3f}    Memory (MB): {:.3f}'.format(t1,mem1))

print('\n Model rankings the same as elimination? - {}'.format(
        np.array_equal(rank1,rank0))
      )
print(' Model p-values the same as elimination? - {}'.format(
        np.array_equal(pVals1,pVals0))
     )
print(' Mean absolute deviation in P-values from elimination: {:.3f}'.format(
        np.mean(np.abs(pVals1-pVals0)))
      )
#------------------------------------------------------------------------------
# Generate plots
fontSize = 32
y_min, y_max = (0,1.05)
x_min, x_max = (0,numMods+5)

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(1, 1, 1)
ax.plot(pVals0,'b', linewidth=1, label = r'Elimination')
ax.plot(pVals2,'k', linewidth=1, label = r'2-pass')
ax.plot(pVals1,'r', linewidth=1, label = r'1-pass w. updating')

ax.legend(loc='upper left', frameon=False, prop={'size':fontSize})
ax.set_ylim(top = y_max, bottom = y_min)
ax.set_xlim(left = x_min,right = x_max)
ax.plot(x_max, y_min, ">k", ms=15, clip_on=False)
ax.plot(x_min, y_max, "^k", ms=15, clip_on=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='x', pad=15, labelsize=fontSize)
ax.tick_params(axis='y', pad=15, labelsize=fontSize)
