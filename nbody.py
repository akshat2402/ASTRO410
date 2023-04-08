
# Import libraries
import numpy as np
import os, sys, pdb
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D
from astropy.stats import LombScargle

# class for celestial body
class Body:
    # define class variables
    G = 6.67259e-8 # cgs
    instances = []

    # initialization w/ properties + ephemeris data
    def __init__(self, name, mass, radius, semimajor, \
                 eccentricity, period, inclination, \
                 ascending_node, long_periapsis, \
                 mean_longitude, random_init = False, *args):
        # set parameters + ephemeris attributes
        self.name = name
        self.mass = mass
        self.radius = radius
        self.semimajor = semimajor
        self.eccentricity = eccentricity
        self.period = period
        self._inclination = np.deg2rad(inclination)
        self._ascending_node = np.deg2rad(ascending_node)
        self._long_periapsis = np.deg2rad(long_periapsis)
        self._mean_longitude = np.deg2rad(mean_longitude)

        # calculate standard gravitational parameter
        if ((self.semimajor != 0) & (self.period != 0)):
            self.mu = (4 * np.pi**2 * self.semimajor**3) / self.period**2
        else:
            self.mu = 0

        # update instances
        self.instances.append(self)

        # set initial conditions
        self.set_init_conditions(random_init)

    # method, set initial position and velocity three vectors
    def set_init_conditions(self, random_init):
        # calculate initial position
        # calculate argument of periapsis and mean anomaly
        self._mean_anomaly = self._mean_longitude - self._long_periapsis
        self._arg_periapsis = self._long_periapsis - self._ascending_node

        # solve the Kepler equation
        kepler = lambda EE, MM, ee: EE - ee * np.sin(EE) - MM
        self._ecc_anomaly = newton(kepler, 175, args=(self._mean_anomaly, self.eccentricity))

        # add phases if random_init flag
        if random_init:
            true_anomaly = np.arccos((np.cos(self._ecc_anomaly) - self.eccentricity)/\
                                     (1 - self.eccentricity * np.cos(self._ecc_anomaly)))
            random_phase = np.deg2rad(360. * np.random.random())
            true_anomaly = true_anomaly + random_phase
            self._ecc_anomaly = np.arccos((self.eccentricity + np.cos(true_anomaly))/\
                                          (1 + self.eccentricity * np.cos(true_anomaly)))

        # convert spatial coordinates from eccentric anomaly
        PP = self.semimajor * (np.cos(self._ecc_anomaly) - self.eccentricity)
        QQ = self.semimajor * np.sin(self._ecc_anomaly) * np.sqrt(1. - self.eccentricity**2)

        # rotate coordinates into 3D coordinate system
        # rotate by argument of periapsis
        xx = np.cos(self._arg_periapsis) * PP - np.sin(self._arg_periapsis) * QQ
        yy = np.sin(self._arg_periapsis) * PP + np.cos(self._arg_periapsis) * QQ

        # rotate by inclination
        zz = np.sin(self._inclination) * xx
        xx = np.cos(self._inclination) * xx

        # rotate by longitude of ascending node
        xt = xx
        xx = np.cos(self._ascending_node) * xt - np.sin(self._ascending_node) * yy
        yy = np.sin(self._ascending_node) * xt + np.cos(self._ascending_node) * yy

        # set initial position
        self.position = np.array([xx, yy, zz])

        # calculate initial velocity
        if ((self.semimajor == 0) & (self.period == 0)):
            self.velocity = np.array([0., 0., 0.])
            self.kinetic = np.array([0.])
            return

        # calculate Ldot and Edot
        Ldot = 2 * np.pi / self.period # Mdot = Ldot
        Edot = Ldot / (1 - self.eccentricity * np.cos(self._ecc_anomaly))

        # convert spatial coordinates from Edot
        Pdot = -self.semimajor * np.sin(self._ecc_anomaly) * Edot
        Qdot = self.semimajor * np.cos(self._ecc_anomaly) * Edot * np.sqrt(1 - self.eccentricity**2)

        # rotate velocities into 3D coordinate system
        # rotate by argument of periapsis
        xdot = np.cos(self._arg_periapsis) * Pdot - np.sin(self._arg_periapsis) * Qdot
        ydot = np.sin(self._arg_periapsis) * Pdot + np.cos(self._arg_periapsis) * Qdot

        # rotate by inclination
        zdot = np.sin(self._inclination) * xdot
        xdot = np.cos(self._inclination) * xdot

        # rotate by longitude of ascending node
        xt = xdot
        xdot = np.cos(self._ascending_node) * xt - np.sin(self._ascending_node) * ydot
        ydot = np.sin(self._ascending_node) * xt + np.cos(self._ascending_node) * ydot

        # set initial velocity
        self.velocity = np.array([xdot, ydot, zdot])

        # calculate initial kinetic energy
        self.kinetic = np.array([0.5 * self.mass * self.velocity.dot(self.velocity)])
        return

    # method, calculate grav. acc. at a specified position
    def acceleration(self, position):
        dr = self.current_position - position
        return self.G * self.mass * dr / (dr.dot(dr))**(3./2.)

    # method, calculate grav. pot. between body and another mass at position
    def grav_potential(self, mass, position):
        dr = self.current_position - position
        return self.G * mass * self.mass / (dr.dot(dr))**(1./2.)

    # method, calculate kinetic energy
    def get_kinetic_energy(self):
        return 0.5 * self.mass * self.current_velocity.dot(self.current_velocity)

    @property
    def current_position(self):
        if self.position.shape == (3,):
            return self.position
        else:
            return self.position[-1]

    @property
    def current_velocity(self):
        if self.velocity.shape == (3,):
            return self.velocity
        else:
            return self.velocity[-1]

    @property
    def position_to_AU(self):
        return self.position * 6.685e-14

    @property
    def semimajor_to_AU(self):
        return self.semimajor * 6.685e-14

# group of celestial bodies
class System:
    # initialization, sort out bodies in system
    def __init__(self, name, *args):
        self.name = name
        self.bodies = [obj for obj in args if isinstance(obj, Body) is True]
        self.number_bodies = len(self.bodies)

    # evolve the system by one time step (dt)
    def step(self, solver, dt):
        self.time_step = dt

        # step each body in the system
        new_pot = 0
        new_kin = 0
        for body in self.bodies:
            # make list of acceleration methods
            grav_funcs = [obj.grav_potential for obj in self.bodies if body is not obj]
            accel_funcs = [obj.acceleration for obj in self.bodies if body is not obj]

            # calculate new position and velocity of body
            newpos, newvel = solver(body.current_position, \
                                    body.current_velocity, \
                                    accel_funcs, dt)

            # append new position and velocity of body
            body.position = np.vstack([body.position, newpos])
            body.velocity = np.vstack([body.velocity, newvel])

            # calculate kinetic energy of body
            body.kinetic = np.vstack([body.kinetic, body.get_kinetic_energy()])

    # plot the xy plane of the system
    def plot_system2D(self, scope = 'outer', show = False, outdir = None,):
        fig, ax = plt.subplots()

        # plot orbits
        rad = []
        for body in self.bodies:
            rad.append(body.semimajor_to_AU)
            pos1 = body.position_to_AU[:, 0]
            pos2 = body.position_to_AU[:, 1]
            ax.plot(pos1, pos2, 'k-', linewidth = 0.5)
            ax.plot(pos1[-1], pos2[-1], 'o', label = body.name)

        # find largest semimajor axis
        if (scope == 'outer'):
            lim = np.max(rad) + 0.05
        if (scope == 'inner'):
            lim = np.median(rad)

        # make figure pretty
        ax.set_xlabel(r'$\Delta x$ (AU)')
        ax.set_ylabel(r'$\Delta y$ (AU)')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(self.name.replace('_', ' '))
        ax.set_aspect('equal', adjustable='box')
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

        # show/write out
        if show:
            plt.show()
        if outdir is not None:
            fig.savefig(outdir + self.name.replace(' ', '') + '_2D.pdf', bbox_inches='tight')

        # close figure
        plt.clf()
        plt.close()

    # animate the xy plane of the system
    def animate_system2D(self, scope = 'outer', outdir = None, show = False):
        # initialize fig, ax objects
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        # initialize line object
        lines = []
        circles = []
        for i in range(self.number_bodies):
            lobj = ax.plot([], [], 'k-', linewidth=0.5)[0]
            cobj = ax.plot([], [], 'o', markersize=3, label = self.bodies[i].name)[0]
            lines.append(lobj)
            circles.append(cobj)

        patches = lines + circles

        nframes = int(len(self.bodies[0].position_to_AU[:,0])/10)

        def init(*args):
            for line in lines:
                line.set_data([],[])
            for circle in circles:
                circle.set_data([],[])
            return patches

        def animate(i, *args):
            if (i % 8 == 0):
                for lnum, line in enumerate(lines):
                    body = self.bodies[lnum]
                    pos1 = body.position_to_AU[0:i, 0]
                    pos2 = body.position_to_AU[0:i, 1]
                    line.set_data([pos1, pos2])
                for cnum, circle in enumerate(circles):
                    body = self.bodies[cnum]
                    pos1 = body.position_to_AU[i, 0]
                    pos2 = body.position_to_AU[i, 1]
                    circle.set_data(pos1, pos2)
            return patches

        # find largest semimajor axis
        rad = []
        for body in self.bodies:
            rad.append(body.semimajor_to_AU)
        if (scope == 'outer'):
            lim = np.max(rad) + 0.05
        if (scope == 'inner'):
            lim = np.median(rad)

        # set fig parameters
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_xlabel(r'$\Delta x$ (AU)')
        ax.set_ylabel(r'$\Delta y$ (AU)')
        ax.set_title(self.name.replace('_', ' '))
        ax.set_aspect('equal', adjustable='box')
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

        # save the animation
        print('Writing .mp4 file (this is slow)')
        anim = animation.FuncAnimation(fig, animate, frames=nframes, blit=True)
        anim.save(outdir + self.name.replace(' ', '') + '_2D.mp4', fps=60, dpi=500)

        plt.clf()
        plt.close()

    def plot_system3D(self, scope = 'outer', show = False, outdir = None):
        # initialize fig, ax objects
        fig = plt.figure(figsize=plt.figaspect(0.9)*1.2)
        ax = fig.add_subplot(111, projection='3d')

        # plot orbits
        rad = []
        for body in self.bodies:
            rad.append(body.semimajor_to_AU)
            pos1 = body.position_to_AU[:, 0]
            pos2 = body.position_to_AU[:, 1]
            pos3 = body.position_to_AU[:, 2]
            ax.plot(pos1, pos2, pos3, 'k-', linewidth = 0.5)
            ax.plot([pos1[-1]], [pos2[-1]], [pos3[-1]], 'o', markersize = 3, label = body.name)

        # find largest semimajor axis
        if (scope == 'outer'):
            lim = np.max(rad) + 0.05
        if (scope == 'inner'):
            lim = np.median(rad)

        # make figure pretty
        ax.set_xlabel(r'$\Delta x$ (AU)')
        ax.set_ylabel(r'$\Delta y$ (AU)')
        ax.set_zlabel(r'$\Delta z$ (AU)')
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(-lim, lim)
        ax.set_title(self.name.replace('_', ' '))
        #ax.set_aspect('equal', adjustable='box')
        ax.legend(bbox_to_anchor=(0.1,0.5), loc="center right")

        # show/write out
        if show:
            plt.show()
        if outdir is not None:
            fig.savefig(outdir + self.name.replace(' ', '') + '_3D.pdf')

        # close figure
        plt.clf()
        plt.close()

    def plot_errors(self, show = False, outdir = None, write = False):
        # initialize fig, ax objects
        fig, ax = plt.subplots()

        # plot the errors
        total = 0
        for body in self.bodies:
            total += body.kinetic
        steps = np.arange(len(total)-2)
        delta = total[2:] - total[1]
        ax.plot(steps, delta/total[1])

        # make figure pretty
        ax.set_xlabel(r'Step Number')
        ax.set_ylabel(r'$\Delta KE / KE$')
        ax.set_title(self.name.replace('_', ' ') + ' - error')

        # show/write out
        if show:
            plt.show()
        if outdir is not None:
            fig.savefig(outdir + self.name.replace(' ', '') + '_errors.pdf', bbox_inches='tight')

        if write:
            np.savetxt('errors.txt', delta/total[1])

        # close figure
        plt.clf()
        plt.close()

    def plot_periodogram(self, show = False, outdir = None):
        # get projected velocities of star
        velx = self.bodies[0].velocity[:,0]
        nsteps = len(velx)
        time = np.arange(0, nsteps) * self.time_step

        # find the most massive planet
        masses = [body.mass for body in self.bodies[1:]]
        massive_ind = np.argmax(masses) + 1

        # set frequency limits around period of massive planet
        minfreq = 0.5 * 1./self.bodies[massive_ind].period
        maxfreq = 2.0 * 1./self.bodies[massive_ind].period

        # calculate the periodogram
        frequency, power = LombScargle(time, velx).autopower(minimum_frequency=minfreq,
                                                             maximum_frequency=maxfreq,
                                                             samples_per_peak=20)
        period = 1./frequency

        # plot the result
        plt.plot(period, power)


        # close figure
        plt.clf()
        plt.close()
