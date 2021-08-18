#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:18:41 2020

Code adapted from GDit (Aaron Dotter)

@author: shashank
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import figure

import arviz as az
import starry

#constants
G=6.67428e-8
sigma=5.67040e-5
f23=2.0/3.0
Lsun=3.8418e33
Msun=1.989e33
Rsun=6.96e10

#ELR11 equations
#gives the value of phi
def eq24(phi,theta,omega,rtw):
    tau = (pow(omega,2) * pow(rtw*np.cos(theta),3) )/3.0 + np.cos(theta) + np.log(np.tan(0.5*theta))
    return np.cos(phi) + np.log(np.tan(0.5*phi)) - tau

#solve for rtw given omega
def eq30(rtw,theta,omega):
    w2=omega*omega
    return (1./w2)*(1./rtw - 1.0) + 0.5*(pow(rtw*np.sin(theta),2) - 1.0)

#ratio of equatorial to polar Teff
def eq32(omega):
    w2=omega*omega
    return np.sqrt(2./(2.+w2))*pow(1.-w2, 1./12.)*np.exp(-(4./3.)*w2/pow(2+w2, 3))


def solve_ELR(omega,theta): #eq.26, 27, 28; solve the ELR11 equations
    """calculates r~, Teff_ratio, and Flux_ratio"""
    #theta is the polar angle.
    #this routine calculates values for 0 <= theta <= pi/2
    #everything else is mapped into this interval by symmetry
    # theta = 0 at the pole(s)
    # theta = pi/2 at the equator
    # -pi/2 < theta < 0: theta -> abs(theta)
    #  pi/2 > theta > pi: theta -> pi - theta
    if np.pi/2 < theta <= np.pi:
        theta = np.pi - theta
    if -np.pi/2 <= theta < 0:
        theta = abs(theta)

    if omega==0.0: #common sense
        return np.ones(3)
    
    else:
        #first we solve equation 30 for rtw
        q = root(fun=eq30,args=(theta, omega), x0=1.0)
        rtw = np.asscalar(q['x'])

        #the following are special solutions for extreme values of theta
        w2r3=pow(omega,2)*pow(rtw,3)
        
        if theta==0.0: #pole, eq. 27
            Fw = np.exp( f23 * w2r3 )

        elif theta==0.5*np.pi: #equator, eq. 28
            if omega < 1.0:
                Fw = pow(1.0 - w2r3, -f23)
            else:
                Fw = 0.0

        else: #general case for Fw
            q = root(fun=eq24,args=(theta, omega, rtw), x0=theta)
            phi = np.asscalar(q['x'])
            
            Fw = pow(np.tan(phi)/np.tan(theta), 2)

        #equation 31 and similar for Fw
        term1 = pow(rtw,-4)
        term2 = pow(omega,4)*pow(rtw*np.sin(theta),2)
        term3 = -2*pow(omega*np.sin(theta),2)/rtw
        gterm = np.sqrt(term1+term2+term3)
        Flux_ratio = Fw*gterm
        Teff_ratio = pow(Flux_ratio,0.25)
        return rtw, Teff_ratio, Flux_ratio
def Rp_div_Re(omega):
    rtw, Teff_ratio, Flux_ratio = solve_ELR(omega, theta=0)
    return rtw   



#plot one spheroid with Teff variation across the surface
def plot_one_spheroid_inc(omega,inc,figname=None,cmap=plt.cm.plasma_r):
    
    n=100
    
    def get_one_F(omega,theta):
        r,T,F=solve_ELR(omega,theta)
        return F
    
    Re=1.0
    Rp=Rp_div_Re(omega)*Re
   
    # Set of all spherical angles:
    phi = np.linspace(-np.pi, np.pi, n)
    nu = np.linspace(-np.pi/2, np.pi/2, n)

    angles=np.outer(np.ones_like(phi),nu)
    F=np.outer(np.ones_like(phi),np.ones_like(nu))
    for j in range(n):
        theta=angles[0,j]
        F[:,j]=get_one_F(omega,theta)
        
    if(Rp==Re): #spherical
        a=Re
        x=a*np.outer(np.cos(phi),np.cos(nu))
        y=a*np.outer(np.sin(phi),np.cos(nu))
        z=a*np.outer(np.ones_like(phi), np.sin(nu))
        
        F=np.outer(np.ones_like(phi), np.ones_like(nu))
        
    else: #spheroidal
        mu=np.arctanh(Rp/Re)
        a=np.sqrt(Re*Re - Rp*Rp)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = a*np.cosh(mu) * np.outer(np.cos(phi), np.cos(nu))
        y = a*np.cosh(mu) * np.outer(np.sin(phi), np.cos(nu))
        z = a*np.sinh(mu) * np.outer(np.ones_like(phi), np.sin(nu))
        Fmin=F.min()
        Fmax=F.max()
        F=(F-Fmin)/(Fmax - Fmin)


    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    ax.view_init(elev=inc,azim=0)
    ax.plot_surface(x,y,z,rstride=1,cstride=1,facecolors=cmap(F))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    #ax.set_axis_off()
    plt.title(r'$\omega$='+str(omega)+', $i_*='+str(90-inc)+r'^{\circ}$')
    if figname is not None:
        plt.savefig(figname,dpi=300)

def plot_star_withLD(omega=.75, inc=90, u1=0.2,u2=0.2,beta=0.23,tpole=8500,wav=800):
    """Fig 1 in the paper"""
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    map = starry.Map(udeg=2, gdeg=4, oblate=True) #ydeg = 2*order_approx udeg=2
    map[1]=u1
    map[2]=u2
    map.omega=omega
    map.beta=beta
    map.wav=wav
    map.tpole=tpole
    map.f = gdark.f(omega)
    map.inc=inc
    plt.title(r'$\omega$='+str(omega)+', $i_*='+str(int(map.inc.eval()))+r'^{\circ}$',fontsize=16)
    map.show(ax=ax)
    plt.savefig("spheroid_"+str(omega)+"_"+str(int(map.inc.eval()))+".pdf")
    
if __name__=='__main__':
    plot_one_spheroid_inc(omega=0.8,inc=0,figname='../imgs/spheroid_90.pdf',cmap=plt.cm.plasma_r)
    
