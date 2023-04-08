#jupyter-notebook: %run Earth_orbit.py
#terminal: python Earth_orbit.py

from pylab import *
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

G = 6.6743e-11 #Gravitation constant
m0 = 1.9891e30 #mass of sun
year = 3.154e7 #seconds in a year

def accel(x):
    r2 = x[0]**2 + x[1]**2
    r3 = r2*sqrt(r2)
    return -G*m0*x/r3

def leapfrog_orbit(x0=1.0, dt=2.5e-2):
    au = 1.496e11
    n = int(2+1./dt)
    t = linspace(0.0, 1.+0.2*dt, n)
    dt = t[1]-t[0]
    x = zeros((n,2))
    v = zeros((n,2))
    x[0] = array([x0*au,0.0])
    v0 = sqrt(G*m0/x[0,0])
    v[0] = array([0.0,v0])
    dt *= year
    dth = dt/2.

    fig,ax = plt.subplots(1, 2, figsize=(8,4), gridspec_kw={'wspace':0.5})
    ax[0].set_aspect('equal')
    ax[0].set_aspect('equal')
    ps,=ax[0].plot([0],[0], color='r', marker='o', markersize=10)
    p0,=ax[0].plot([0],[0], color='k')
    pp,=ax[0].plot([1],[0], color='b', marker='o')
    ax[0].set_xlim(-1.2,1.2)
    ax[0].set_ylim(-1.2,1.2)
    ax[0].set_xlabel('X (AU)')
    ax[0].set_ylabel('Y (AU)')
    p1,=ax[1].plot([0],[0], color='k')
    ax[1].set_xlim(-0.1,1.1)
    ax[1].set_ylim(-2e-2,1e-2)
    ax[1].set_xlabel('Time (Year)')
    ax[1].set_ylabel(r'$\Delta E/E$ (%)')
    et = zeros(n)
    et[0] = 0.5*sum(v[0]**2)-G*m0/x[0,0]
    
    for i in range(1,n):
        a = accel(x[i-1])
        vh = v[i-1] + a*dth
        x[i] = x[i-1]+vh*dt
        a = accel(x[i])
        v[i] = vh + a*dth
        et[i] = 0.5*sum(v[i]**2) - G*m0/sqrt(sum(x[i]**2))
        p0.set_data(x[:i+1,0]/au,x[:i+1,1]/au)
        pp.set_data([x[i,0]/au], [x[i,1]/au])
        p1.set_data(t[:i+1], 100*(et[:i+1]/et[0]-1))
        display(fig)
        clear_output(wait=True)
        plt.pause(0.05)

leapfrog_orbit()
