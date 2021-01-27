#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 23:33:16 2020

@author: shashank
"""

from pytransit import OblateStarModel, QuadraticModel
import pandas as pd
import lightkurve as lk
import numpy as np
tmo = OblateStarModel(sres=1000, pres=10)
tmc = QuadraticModel(interpolate=False)


df = pd.read_csv("/Users/shashank/Documents/Python Files/wasp33TESS_clean.csv")
lc = lk.LightCurve(time=df.time,flux=df.flux)
tmo.set_data(lc.time)

G_mks = 6.67e-11
Msun = 1.989e+30
Rsun = 6.95700e8
M_star = 1.5
R_star = 1.464
u1 = 0.246
u2 = 0.252

omega_s = 0.2/(np.sin(np.radians(180+inc)))
    
k = np.array([0.10714]) #Rprs
p_rot = (2*np.pi*(R_star*Rsun)**(3/2))/(omega_s*(G_mks*M_star*Msun)**(1/2)*(60*60*24))
print(p_rot) #rotation period in days
t0, p, a, i, az, e, w = 0.0, 1.2198696, 3.605, 88.01*(np.pi/180), 92*(np.pi/180.), 0.0, 0.0 #real t0 is 2458792.63403
rho, rperiod, tpole, phi, beta = 0.59, p_rot,7400, (inc)*(np.pi/180.), 0.22
#rho, rperiod, tpole, phi, beta = 0.59, 100,7400, -50.0*(np.pi/180.), 0.22
ldc = np.array([u1, u2]) # Quadtratic limb darkening coefficients

flux_om = tmo.evaluate_ps(k, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, az, e, w)