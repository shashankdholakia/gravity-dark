#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:29:45 2021

@author: shashank
"""

import numpy as np
import matplotlib.pyplot as plt

import utils
import starry

from scipy.integrate import quad
from scipy.special import hyp2f1, ellipeinc
from scipy.special import comb

def test_analytic(plot=False):
    tol = 1e-12
    dtol = 1e-5
    
    # Settings
    lmax = 8
    r = 0.25
    theta = 0.4
    b = 0.1
    phi1 =  0
    phi2 = np.pi
    xi1 = 0
    xi2 = 0

    df = 1e-8
    solver = utils.Solver(lmax)
    
    # Compute for f = 0
    sT = solver.get_sT(b, r, 0.24, theta, phi1, phi2, xi1, xi2).reshape(-1)
    sTnum = utils.sT_numerical(lmax, [phi1, phi2], [xi1, xi2], b, r, 0.24, theta).reshape(-1)
    
    # Compute for f = 0 + df to get the deriv
    sT_df = solver.get_sT(b, r, df, theta, phi1, phi2, xi1, xi2).reshape(-1)
    dsTdf = (sT_df - sT) / df
    sTnum_df = utils.sT_numerical(lmax, [phi1, phi2], [xi1, xi2], b, r, df, theta).reshape(-1)
    dsTnumdf = (sTnum_df - sTnum) / df
    
    
    
    if plot==True:
        # Compare
        fig, ax = plt.subplots(2,1, figsize=(12, 7), sharex=True)
        ax[0].plot(np.abs(sT), lw=2, label="analytic")
        ax[0].plot(np.abs(sTnum), lw=1, label="numerical")
        bad = np.array(np.abs(sTnum))
        bad[np.abs(sT - sTnum) / np.abs(sT) < tol] = np.nan
        for n in range((lmax + 1) ** 2):
            ax[0].plot(n, bad[n], color="k", marker=r"${:.0f}$".format(solver.get_case(n)), ms=10)
        ax[0].legend(loc="upper right", fontsize=10)
        ax[0].set_ylabel(r"$s^\top$", fontsize=18)
        ax[0].set_yscale("log")
        
        ax[1].plot(np.abs(dsTdf), lw=2, label="analytic")
        ax[1].plot(np.abs(dsTnumdf), lw=1, label="numerical")
        bad = np.array(np.abs(dsTdf))
        bad[(np.abs(dsTdf) < tol) | (np.abs(dsTdf - dsTnumdf) / (np.abs(dsTdf) + tol) < dtol)] = np.nan
        for n in range((lmax + 1) ** 2):
            ax[1].plot(n, bad[n], color="k", marker=r"${:.0f}$".format(solver.get_case(n)), ms=10)
        ax[1].legend(loc="upper right", fontsize=10)
        ax[1].set_ylabel(r"$\frac{\mathrm{d}s^\top}{\mathrm{d}f}$", fontsize=18)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("coefficient");