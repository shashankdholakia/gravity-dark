#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:31:25 2021

@author: shashank
"""

import numpy as np
from scipy.constants import h, c
from scipy.constants import k as kb

import starry
from scipy.spatial.transform import Rotation as R
starry.config.lazy = False
starry.config.quiet = True

try:
    from numba import jit, njit
except ImportError:
    print("No numba, falling back on slower version")
    jit = lambda x: x # return the function itself

@njit
def f_eff(omega, i_star):
    inc = np.radians(i_star)
    return 1 - np.sqrt((1-f(omega))**2 * np.cos(inc)**2 + np.sin(inc)**2)


@njit
def f(omega):
    return 1- 2/(omega**2 + 2)
@njit
def f_bouvier(omega):
    return 1 - ((2/3)**(-3/2) * omega) / (3*np.sin(np.arcsin((2/3)**(-3/2) * omega) / 3))

@njit
def planck(l, t):
    return 2.*h*c**2/l**5/(np.exp(h*c/(l*kb*t)) - 1.)

@njit
def flux_oblate(x, y, wav, omega, beta, tpole):
    a= (1.-x**2-y**2)
    b = (1.-f(omega))
    
    temp = tpole * b**(2*beta) * ((-a*b**2 + (a-1)*(-(omega**2)*(a*b**2 - a + 1)**(3./2.) + 1.)**2.) / (-a*b**2 + a - 1)**3)**(beta/2)
    return planck(wav, temp)



@njit
def flux_spherical(x, y, wav, omega, beta, tpole):
    
    b = (1.-f(omega))
    temp = tpole * ((omega**4 - 2*omega**2)*(x**2 + y**2) + 1.)**(beta/2)
    return planck(wav, temp)


def plot_precision_long(omegas):
    ref_fluxes = []
    sph_harm_flux = []
    map0 = starry.Map(ydeg=3)
    lat, lon, Y2P, P2Y, Dx, Dy = map0.get_pixel_transforms(oversample=10)
    phi = np.radians(lat)
    theta = np.radians(lon)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    plt.clf()
    for omega in omegas:
        #first pixelize a map to find reference values

        #reference flux
        flux = flux_oblate(x, y, 5.1e-7, omega, 0.22, 7600)
        flux /= np.max(flux)
        ref_fluxes.append(flux)
        
        #now compare to spherical harmonic transform
        yarr = P2Y.dot(flux)
        p = Y2P.dot(yarr)
        sph_harm_flux.append(p)
        plt.scatter(lat[lat.argsort()], ((flux-p)/flux)[lat.argsort()], label=r"$\omega=$"+ str(omega))
        
    plt.legend()
    #plt.yscale('log')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Residual')
    plt.show()
    
    

if __name__=='__main__':
    plot_precision_long([0.2,0.4,0.6,0.8])
        


