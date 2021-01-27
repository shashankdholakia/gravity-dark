#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:31:25 2021

@author: shashank
"""

import numpy as np
from scipy.constants import h, c
from scipy.constants import k as kb

try:
    from numba import jit, njit
except ImportError:
    print("No numba, falling back on slower version")
    jit = lambda x: x # return the function itself


@njit
def f(omega):
    return 1 - ((2/3)**(-3/2) * omega) / (3*np.sin(np.arcsin((2/3)**(-3/2) * omega) / 3))

@njit
def planck(l, t):
    return 2.*h*c**2/l**5/(np.exp(h*c/(l*kb*t)) - 1.)

@njit
def flux_oblate(x, y, wav, omega, beta, tpole):
    a= (1.-x**2-y**2)
    b = (1.-f(omega))
    
    temp = tpole * b**(2*beta) * ((-a*b**2 + (a-1)*(-(omega**2)*(a*b**2 - a + 1)**(3./2.) + 1.)**2.) / (-a*b**2 + a - 1)**3)**(beta/2)
    print(temp)
    return planck(wav, temp)









