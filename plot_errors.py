

# Import libraries
import numpy as np
import pandas as pd
import os, sys, pdb
import matplotlib.pyplot as plt


# read in errors
rk4 = np.loadtxt('errors_rk4.txt')
leap = np.loadtxt('errors_leapfrog.txt')
ruth = np.loadtxt('errors_ruth.txt')

nstep = np.arange(0, len(rk4))

# plot the errors
fig, ax = plt.subplots()

ax.plot(nstep, rk4, linestyle = '--', label='RK4')
ax.plot(nstep, leap, linestyle = '-.', label='Leapfrog')
ax.plot(nstep, ruth, linestyle = ':', label='Ruth')

ax.set_xlabel('Step Number')
ax.set_ylabel(r'$\Delta KE / KE$')

ax.legend()

fig.savefig('all_errors.pdf', bbox_inches='tight')
