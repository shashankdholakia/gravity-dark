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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib.animation import FFMpegWriter
import pandas as pd
import lightkurve as lk


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

def plot_star_withLD(omega=.75, inc=90, obl=0,u1=0.2,u2=0.2,beta=0.23,tpole=8500,wav=800, anim=False):
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
    map.f = 1-2/(omega**2 + 2)
    map.inc=inc
    map.obl=obl
    if anim==False:
        plt.title(r'$\omega$='+str(omega)+', $i_*='+str(int(map.inc.eval()))+r'^{\circ}$',fontsize=16)
        map.show(ax=ax)
        plt.savefig("spheroid_"+str(omega)+"_"+str(int(map.inc.eval()))+".pdf")
    else:
        theta = np.linspace(0, 360, 50)
        map.show(ax=ax,theta=theta,file="spheroid_"+str(omega)+"_"+str(int(map.inc.eval()))+".mp4", bitrate=10000, dpi=300)
    

def plot_system_withLC(
        sys,
        tstart,
        tend,
        frames,
        cmap="plasma",
        res=300,
        interval=50,
        file=None,
        figsize=(3, 3),
        html5_video=True,
        window_pad=1.0,
        supersample=10,
        plot_lotlan_lines = False,
        data_time=None, 
        data_flux=None,
        bitrate=-1):
    
    t = np.linspace(tstart,tend,frames)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    plt.rcParams['animation.convert_path'] = r'/usr/bin/convert'
    img_pri, img_sec, x, y, z = sys.ops.render(t,
                                           res,
                                           sys.primary._r,
                                           sys.primary._m,
                                           sys.primary._prot,
                                           sys.primary._t0,
                                           sys.primary._theta0,
                                           sys.primary._map._inc,
                                           sys.primary._map._obl,
                                           sys.primary._map.fproj.eval(),
                                           sys.primary._map._y.eval(),
                                           sys.primary._map._u.eval(),
                                           sys.primary._map._f.eval(),
                                           [i._r for i in sys.secondaries],
                                           [i._m for i in sys.secondaries],
                                           [i._prot for i in sys.secondaries],
                                           [i._t0 for i in sys.secondaries],
                                           [i._theta0 for i in sys.secondaries],
                                           [i._porb for i in sys.secondaries],
                                           [i._ecc for i in sys.secondaries],
                                           [i._w for i in sys.secondaries], 
                                           [i._Omega for i in sys.secondaries],
                                           [i._inc for i in sys.secondaries],
                                           [i.map._inc for i in sys.secondaries], 
                                           [i.map._obl for i in sys.secondaries], 
                                           [i.map._y.eval() for i in sys.secondaries], 
                                           [i.map._u.eval() for i in sys.secondaries], 
                                           [i.map._f.eval() for i in sys.secondaries], 
                                           [0 for i in sys.secondaries],
                                          )
    
    img_pri = img_pri.eval()
    img_sec = img_sec.eval()
    x, y, z = x.T.eval()/sys.primary._r.eval(), y.T.eval()/sys.primary._r.eval(), z.T.eval()/sys.primary._r.eval()
    
    r = [sec._r.eval()/sys.primary._r.eval() for sec in sys.secondaries]
    
    # Set up the plot
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 2]})

    ax.axis("off")
    ax.set_xlim(-1.0 - window_pad, 1.0 + window_pad)
    ax.set_ylim(-1.0 - window_pad, 1.0 + window_pad)
    
    # Ensure we have an array of frames
    if len(img_pri.shape) == 3:
        nframes = img_pri.shape[0]
    else:  # pragma: no cover
        nframes = 1
        img_pri = np.reshape(img_pri, (1,) + img_pri.shape)
        img_sec = np.reshape(img_sec, (1,) + img_sec.shape)
    animated = nframes > 1

    # Render the first frame
    img = [None for n in range(1 + len(sys.secondaries))]
    circ = [None for n in range(1 + len(sys.secondaries))]
    extent = np.array([-1.0, 1.0, -1.0, 1.0])
    img[0] = ax.imshow(
        img_pri[0],
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="antialiased",
        vmin=np.nanmin(img_pri),
        vmax=np.nanmax(img_pri),
        animated=animated,
        zorder=0.0,
    )
    
    if plot_lotlan_lines:
        color='white'
    else:
        color='white'
    circ[0] = Ellipse(
        (0, 0),
        2,
        2 * (1 - sys.primary._map.fproj.eval()),
        angle= 180/np.pi * sys.primary.map._obl.eval(),
        color=color,
        fill=False,
        zorder=1e-2,
        lw=2,
    )

    ax.add_artist(circ[0])
    
    for i, _ in enumerate(sys.secondaries):
        extent = np.array([x[i, 0], x[i, 0], y[i, 0], y[i, 0]]) + (
            r[i] * np.array([-1.0, 1.0, -1.0, 1.0])
        )
        img[i + 1] = ax.imshow(
            img_sec[i, 0],
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="antialiased",
            vmin=np.nanmin(img_sec),
            vmax=np.nanmax(img_sec),
            animated=animated,
            zorder=z[i, 0],
        )
        circ[i] = plt.Circle(
            (x[i, 0], y[i, 0]),
            r[i],
            color="k",
            fill=True, #change if the planet has a map
            zorder=z[i, 0] + 1e-3,
            lw=0,
        )
        ax.add_artist(circ[i])
        


    #plot the lightcurve in light grey
    t_super = np.linspace(tstart,tend,frames*supersample)
    flux = sys.flux(t_super, integrated=True).eval()
    ax1.plot(t_super,flux,lw=1,alpha=0.5,c='k')
    line, = ax1.plot(t_super[0],flux[0], lw=3,c='k',solid_capstyle='round')
    ax1.set_xlabel("Time",fontsize=12)
    ax1.set_ylabel("Brightness",fontsize=12)
    if data_time is not None and data_flux is not None:
        ax1.scatter(data_time, data_flux, s=5,c='black',alpha=0.7,zorder=0)
        line, = ax1.plot(t_super[0],flux[0], lw=3,c='red',solid_capstyle='round',zorder=2)
    ax1.xaxis.set_ticks([])
    ax1.yaxis.set_ticks([])
    
    if plot_lotlan_lines:
        lats = sys.primary.map._get_ortho_latitude_lines(inc=sys.primary.map._inc.eval(), obl=sys.primary.map._obl.eval())
        latlines = [None for n in lats]
        for n, l in enumerate(lats):
            (latlines[n],) = ax.plot(
                    l[0], l[1], "k-", lw=0.5, alpha=0.1, zorder=1e-3
            )
        lons = sys.primary.map._get_ortho_longitude_lines(inc=sys.primary.map._inc.eval(), obl=sys.primary.map._obl.eval())
        lonlines = [None for n in lons]
        for n, l in enumerate(lons):
            (lonlines[n],) = ax.plot(
                    l[0], l[1], "k-", lw=0.5, alpha=0.3, zorder=1e-3
            )
            
    # Animation
    if animated:

        def updatefig(k):

            # Update Primary map
            img[0].set_array(img_pri[k])
            line.set_data(t_super[0:(k+1)*supersample],flux[0:(k+1)*supersample])
            
            lats = sys.primary.map._get_ortho_latitude_lines(inc=sys.primary.map._inc.eval(), obl=sys.primary.map._obl.eval())
            lons = sys.primary.map._get_ortho_longitude_lines(inc=sys.primary.map._inc.eval(), obl=sys.primary.map._obl.eval())
            for n, l in enumerate(lats):
                latlines[n].set_data(l[0], l[1])
            for n, l in enumerate(lons):
                lonlines[n].set_data(l[0], l[1])

            # Update Secondary maps & positions
            for i, _ in enumerate(sys.secondaries):
                extent = np.array([x[i, k], x[i, k], y[i, k], y[i, k]]) + (
                    r[i] * np.array([-1.0, 1.0, -1.0, 1.0])
                )
                if np.any(np.abs(extent) < 1.0 + window_pad):
                    img[i + 1].set_array(img_sec[i, k])
                    img[i + 1].set_extent(extent)
                    img[i + 1].set_zorder(z[i, k])
                    circ[i].center = (x[i, k], y[i, k])
                    circ[i].set_zorder(z[i, k] + 1e-3)
                
            theta=sys.primary._theta0.eval()+t[k]*360./sys.primary._prot.eval()
            
            print(t[k], sys.primary._prot.eval(), sys.primary._theta0.eval()+t[k]*360./sys.primary._prot.eval())
            lons = sys.primary.map._get_ortho_longitude_lines(inc=sys.primary.map._inc.eval(), obl=sys.primary.map._obl.eval(),theta=theta)
            for n, l in enumerate(lons):
                lonlines[n].set_data(l[0], l[1])

            return img + circ

        ani = FuncAnimation(
            fig, updatefig, interval=interval, blit=False, frames=nframes
        )
        fig.tight_layout()
        # Business as usual
        if (file is not None) and (file != ""):
            if file.endswith(".mp4"):
                #writer = FFMpegWriter(codec="h264", bitrate=bitrate, extra_args=["-preset","-crf","0"]);
                ani.save(file, writer="ffmpeg",bitrate=bitrate)
            elif file.endswith(".gif"):
                ani.save(file, writer="imagemagick", bitrate=bitrate)
            else:  # pragma: no cover
                # Try and see what happens!
                ani.save(file, bitrate=bitrate)
            plt.close()
        else:  # pragma: no cover
            try:
                if "zmqshell" in str(type(get_ipython())):
                    plt.close()
                    if html5_video:
                        display(HTML(ani.to_html5_video()))
                    else:
                        display(HTML(ani.to_jshtml()))
                else:
                    raise NameError("")
            except NameError:
                plt.show()
                plt.close()
                
def plot_system(omega=0.3124822, inc=50.5002724, obl=30, u1=0.2121911, u2=0.18045997, 
                beta=0.23, tpole=10170, wav=800, mstar=2.61610935 , 
                rstar=2.45384724, period=1.4811235, omega_p=-69.1309824+30, rplanet=0.07903191, inc_p = 94.73949318, 
                tstart=-0.15,tend=0.15,frames=100):
    """
    WASP 33: plot_system(omega=.209, inc=69.77, obl=30, u1=0.209, u2=0.217, 
                beta=0.23, tpole=7430, wav=786.5, mstar=1.59 , 
                rstar=1.561, period=1.2198681, omega_p=-109+30, rplanet=0.109, inc_p = 91.196, 
                tstart=-0.08,tend=0.08,frames=100)
    
    KELT 9: 
        
    """
    map = starry.Map(udeg=2, gdeg=4, oblate=True) #ydeg = 2*order_approx udeg=2
    map[1]=u1
    map[2]=u2
    map.omega=omega
    map.beta=beta
    map.wav=wav
    map.tpole=tpole
    map.f = 1-2/(omega**2 + 2)
    map.inc=inc
    map.obl=obl
    G_mks = 6.67e-11
    Msun = 1.989e+30
    Rsun = 6.95700e8
    
    p_rot = (2*np.pi*(rstar*Rsun)**(3/2))/(omega*(G_mks*mstar*Msun)**(1/2)*(60*60*24))

    
    star = starry.Primary(map, m=mstar, r=rstar,prot=p_rot)
    planet = starry.Secondary(
        starry.Map(amp=0),
        porb=period,
        m=0,
        r=rplanet * rstar,
        Omega=omega_p,
        t0=0,
        inc=inc_p,
    )
    
    df = pd.read_csv("/Users/shashank/Documents/Python Files/Kelt9TESS_clean.csv")
    df = df.dropna()
    lc = lk.LightCurve(time=df.time,flux=df.flux, flux_err=df.flux_err)
    params = {'Epoch':1683.4449, 'period':1.4811235,'tdur':0.16361}
    
    #fold the data for plotting
    fold = lc.fold(params['period'],params['Epoch'])
    time = (fold.phase*params['period'])[10000:-10000]
    flux = fold.flux[10000:-10000]
    ferr = fold.flux_err[10000:-10000]
    
    #bin the resulting folded light curve
    binlc = lk.LightCurve(time=time,flux=flux).bin(binsize=12,method='mean')
    
    
    sys = starry.System(star, planet)
    plot_system_withLC(sys, tstart, tend, frames,
                       figsize=(5, 5), window_pad=0.5,file='/Users/shashank/Downloads/kelt9.mp4',
                       res=1000,supersample=10,plot_lotlan_lines=True, bitrate=10000,
                       data_time=binlc.time, data_flux=binlc.flux)
    
if __name__=='__main__':
    #plot_one_spheroid_inc(omega=0.8,inc=0,figname='../imgs/spheroid_90.pdf',cmap=plt.cm.plasma_r)
    #plot_system()
    plot_star_withLD(omega=.75, inc=60, obl=30,u1=0.15,u2=0.15,beta=0.23,tpole=8500,wav=500, anim=True)
    
