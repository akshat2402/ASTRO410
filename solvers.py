

# Import libraries
import numpy as np
import os, sys, pdb
import time

def rk4(position, velocity, funcs, dt):
    # calculate the RK terms
    k1 = velocity
    k1v = 0
    for func in funcs:
        k1v += func(position)

    k2 = velocity + 0.5 * dt * k1v
    k2v = 0
    for func in funcs:
        k2v += func(position + 0.5 * dt * k1)

    k3 = velocity + 0.5 * dt * k2v
    k3v = 0
    for func in funcs:
        k3v += func(position + 0.5 * dt * k2)

    k4 = velocity + 0.5 * dt * k3v
    k4v = 0
    for func in funcs:
        k4v += func(position + 0.5 * dt * k3)

    # calculate new position and velocity
    newpos = position + (dt/6.) * (k1 + 2*k2 + 2*k3 + k4)
    newvel = velocity + (dt/6.) * (k1v + 2*k2v + 2*k3v + k4v)
    return newpos, newvel

def leapfrog(position, velocity, funcs, dt):
    # calculate acceleration
    acc1 = np.zeros(3,)
    for func in funcs:
        acc1 += func(position)

    # calculate new position
    newpos = position + velocity * dt + 0.5 * acc1 * dt**2

    # calculate acceration at new position
    acc2 = np.zeros(3,)
    for func in funcs:
        acc2 += func(newpos)

    # calculate new velocity
    newvel = velocity + 0.5 * (acc1 + acc2) * dt
    return newpos, newvel

def leapfrog_kdk(position, velocity, funcs, dt):
    # calculate accelerations
    acc1 = np.zeros(3,)
    for func in funcs:
        acc1 += func(position)

    # calculate half-step velocity
    vhalf = velocity + 0.5 * acc1 * dt

    # calculate new position
    newpos = position + vhalf * dt

    # calculate acceleration at new position
    acc2 = np.zeros(3,)
    for func in funcs:
        acc2 += func(newpos)

    # calculate new velocity
    newvel = vhalf + 0.5 * acc2 * dt
    return newpos, newvel

def leapfrog_dkd(position, velocity, funcs, dt):
    # calculate half-step position
    phalf = position + 0.5 * velocity * dt

    # calculate acceleration at half-step position
    acc1 = np.zeros(3,)
    for func in funcs:
        acc1 += func(phalf)

    # calculate new velocity and position
    newvel = velocity + acc1 * dt
    newpos = phalf + 0.5 * newvel * dt
    return newpos, newvel

def ruth(position, velocity, funcs, dt):
    # define coefficients
    c1 = 0.67560359598
    c2 = -0.17560359598
    c3 = -0.17560359598
    c4 = 0.67560359598

    d1 = 1.35120719196
    d2 = -1.70241438392
    d3 = 1.35120719196
    d4 = 0.

    # calculate acceleration
    acc1 = np.zeros(3,)
    for func in funcs:
        acc1 += func(position)

    # 1st iteration of velocity and position
    v1 = velocity + c1 * acc1 * dt
    x1 = position + d1 * v1 * dt

    # calculate acceleration
    acc2 = np.zeros(3,)
    for func in funcs:
        acc2 += func(x1)

    # 2nd iteration of velocity and position
    v2 = v1 + c2 * acc2 * dt
    x2 = x1 + d2 * v2 * dt

    # calculate acceleration
    acc3 = np.zeros(3,)
    for func in funcs:
        acc3 += func(x2)

    # 3rd iteration of velocity and position
    v3 = v2 + c3 * acc3 * dt
    x3 = x2 + d3 * v3 * dt

    # calculate acceleration
    acc4 = np.zeros(3,)
    for func in funcs:
        acc4 += func(x3)

    # 4th iteration of velocity and position
    v4 = v3 + c4 * acc4 * dt
    x4 = x3 + d4 * v4 * dt

    return x4, v4
