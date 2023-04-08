# %run kepler.py solar_system.csv leapfrog 72000 1 --show

# Import libraries
import numpy as np
import pandas as pd
import os, sys, pdb
import argparse, time
import matplotlib.pyplot as plt
import solvers
from nbody import *

def main():
    # initialize argparser
    parser = argparse.ArgumentParser(description='Perform N-body simulation of planetary system',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # add command line arguments
    parser.add_argument('system', type=str, help='name of CSV file with system properties')
    parser.add_argument('solver', type=str, help='solver to use: \n'
                                                 'rk4           : 4th-order Runge-Kutta \n'
                                                 'leapfrog      : Leapfrog (symplectic) \n'
                                                 'leapfrog_kdk  : Kick-Drift-Kick \n'
                                                 'leapfrog_dkd  : Drift-Kick-Drift \n'
                                                 'ruth          : 4th-order symplectic \n')
    parser.add_argument('time_step', type=float, help='temporal resolution in seconds')
    parser.add_argument('total_time', type=float, help='total time to simulate in years')
    parser.add_argument('outdir', type=str, nargs='?', default=os.getcwd()+'/', help='optional, directory to write figures to')
    parser.add_argument('--show', action='store_true', help='flag, interactively show plots')
    parser.add_argument('--random', action='store_true', help='flag, randomize initial planet phases')
    parser.add_argument('--animate', action='store_true', help='flag, save system animation')

    # parse the command line arguments
    args = parser.parse_args()
    system = args.system
    solver = args.solver
    time_step = args.time_step
    total_time = args.total_time
    outdir = args.outdir
    show = args.show
    random_init = args.random
    animate = args.animate

    # match the solver argument to function
    try:
        solver_to_call = getattr(solvers, solver)
    except AttributeError:
        sys.exit('Error! Unrecognized solver')

    # read in system data
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    systemdata = pd.read_csv(dir_path + system, comment='#')

    # create solar system Body objects
    bodies = []
    for ind, name in enumerate(systemdata['name'].tolist()):
        bodies.append(Body(*systemdata.iloc[ind].tolist(), random_init=random_init))

    # set parameters for integration
    dt = time_step
    years = total_time
    nstep = int(years * 3.154e7/dt)

    # create System object & evolve in time
    name = os.path.splitext(system)[0]
    SolSystem = System(name, *bodies)
    for i in range(nstep):
        SolSystem.step(solver_to_call, dt)

    # set extent of system to plot
    if (system == 'solar_system.csv'):
        scope = 'inner'
    else:
        scope = 'outer'

    # plot the system
    SolSystem.plot_system2D(scope=scope, outdir=outdir, show=show)
    SolSystem.plot_system3D(scope=scope, outdir=outdir, show=show)
    SolSystem.plot_errors(outdir=outdir, show=show, write=False)

    # animate system if --animation
    if animate: SolSystem.animate_system2D(scope=scope, outdir=outdir)

# run main
if __name__ == '__main__':
    main()
