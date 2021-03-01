#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:35:25 2021

@author: shashank
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.integrate import quad, fixed_quad, romberg

###############################################################################
#GREENS THEOREM AND OBLATE STAR INTEGRATION
    
    
def G(n, f):
    """
    Return the anti-exterior derivative of the nth term of the Green's basis.
    This is a two-dimensional (Gx, Gy) vector of functions of x and y.
    
    
    """

    # Get the mu, nu indices
    l = int(np.floor(np.sqrt(n)))
    m = n - l * l - l
    mu = l - m
    nu = l + m

    # NOTE: The abs prevents NaNs when the argument of the sqrt is
    # zero but floating point error causes it to be ~ -eps.
    z = lambda x, y: np.maximum(1e-12, np.sqrt(np.abs(1 - x ** 2 - (y/(1-f)) ** 2)))

    if nu % 2 == 0:
        
        G = [lambda x, y: 0, lambda x, y: x ** (0.5 * (mu + 2)) * (y/(1-f)) ** (0.5 * nu)]
    
    elif (l == 1) and (m == 0):

        def G0(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return -0.5 * y
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * (-y)

        def G1(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return 0.5 * x
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * x

        G = [G0, G1]

    elif (mu == 1) and (l % 2 == 0):
        
        G = [lambda x, y: (1-f) * x ** (l - 2) * z(x, y) ** 3, lambda x, y: 0]
    
    elif (mu == 1) and (l % 2 != 0):
        
        G = [lambda x, y: x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0]
    
    else:
        
        G = [
            lambda x, y: 0,
            lambda x, y: x ** (0.5 * (mu - 3))
            * (y / (1-f)) ** (0.5 * (nu - 1))
            * z(x, y) ** 3,
        ]
        
    return G

def g(x, y, z=None, n=0):
    """
    Return the nth term of the Green's basis (a scalar).
    
    """
    if z is None:
        z = np.sqrt(1 - x ** 2 - y ** 2)
    l = int(np.floor(np.sqrt(n)))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if (nu % 2 == 0):
        I = [mu // 2]
        J = [nu // 2]
        K = [0]
        C = [(mu + 2) // 2]
    elif (l == 1) and (m == 0):
        I = [0]
        J = [0]
        K = [1]
        C = [1]
    elif (mu == 1) and (l % 2 == 0):
        I = [l - 2]
        J = [1]
        K = [1]
        C = [3]
    elif (mu == 1):
        I = [l - 3, l - 1, l - 3]
        J = [0, 0, 2]
        K = [1, 1, 1]
        C = [-1, 1, 4]
    else:
        I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
        J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
        K = [1, 1, 1]
        C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]
    res = 0
    for i, j, k, c in zip(I, J, K, C):
        res += c * x ** i * y ** j * z ** k
    return res

def primitive(x, y, dx, dy, theta1, theta2, b, rot, n=0):
    """A general primitive integral computed numerically."""

    def func(theta):
        f = 1-b
        Gx, Gy = G(n, f)
        
        xr = lambda theta: x(theta) * np.cos(rot) + y(theta) * np.sin(rot)
        yr = lambda theta: -x(theta) * np.sin(rot) + y(theta) * np.cos(rot)       

        dxr = lambda theta: dx(theta) * np.cos(rot) + dy(theta) * np.sin(rot)
        dyr = lambda theta: -dx(theta) * np.sin(rot) + dy(theta) * np.cos(rot)

        return Gx(xr(theta), yr(theta)) * dxr(theta) + Gy(xr(theta), yr(theta)) * dyr(theta)

    res, _ = quad(func, theta1, theta2, epsabs=1e-12, epsrel=1e-12)
    return res



def tT_numerical(xi, b, theta, n=0):
    """Compute the tT integral numerically from its integral definition."""
    res = 0
    for xi1, xi2 in xi.reshape(-1, 2):
        x = lambda xi: np.cos(theta) * np.cos(xi) - b * np.sin(theta) * np.sin(xi)
        y = lambda xi: np.sin(theta) * np.cos(xi) + b * np.cos(theta) * np.sin(xi)
                
        dx = lambda xi: -np.cos(theta) * np.sin(xi) - b * np.sin(theta) * np.cos(xi)
        dy = lambda xi: -np.sin(theta) * np.sin(xi) + b * np.cos(theta) * np.cos(xi)
        res += primitive(x, y, dx, dy, xi1, xi2, b, theta, n)
    return res

def pT(phi, bo, ro, b, theta, n=0):
    """Compute the pT integral numerically from its integral definition."""
    res = 0
    for phi1, phi2 in phi.reshape(-1, 2):
        x = lambda phi: ro * np.cos(phi)
        y = lambda phi: bo + ro * np.sin(phi)
        
        
        dx = lambda phi: -ro * np.sin(phi)
        dy = lambda phi: ro * np.cos(phi)
                
        res += primitive(x, y, dx, dy, phi1, phi2, b, theta, n)
    return res

def sT(phi, xi, f, theta, b0, rp, n=0):
    """The solution vector for occultations, computed via Green's theorem."""
    b=1-f
    return pT(phi, b0, rp, b, theta, n) + tT_numerical(xi, b, theta, n)

def sT_bruteforce(f, theta, bo, ro, n=0, res=300):

    b=1-f
    # Create a fine grid
    pts = np.linspace(-1, 1, res)
    x, y = np.meshgrid(pts, pts)

    # Check if a point is inside the ellipse of the star
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    in_star = (xr ** 2) + (yr ** 2)/b**2 <= 1

    # Check if a point is under the occultor
    under_occultor = x ** 2 + (y - bo) ** 2 <= ro ** 2

    
    # Now, compute the nth term in the basis...
    term = g(xr, yr/(1-f), n=n)
    
    # And sum it over the integration region
    # Each pixel has area dA = (area of square) / (number of pixels)
    dA = 4 / (x.shape[0] * x.shape[1])
    integral = np.sum(term[in_star & under_occultor] * dA)
    
    return xr, yr, term, under_occultor, in_star, integral


def semi_analytic_sT(f, theta, bo, ro, terms):
    semi_analytic = np.zeros(terms)
    xi = compute_xi(bo, ro, f, theta)
    phi = compute_phi(bo, ro, f, theta)
    xi_2pi = np.sort(np.mod(xi+np.pi, 2*np.pi)-np.pi)
    phi_2pi = np.pi/2 + np.sort(np.mod(phi-np.pi/2, 2*np.pi))
    phi = compute_phi(bo, ro, f, theta)
    try:
        assert len(phi) != 0
    except:
        xr = bo * np.sin(theta)
        yr = bo * np.cos(theta)
        in_star = (xr ** 2) + (yr ** 2)/(1-f)**2 <= 1
        if in_star:
            phi_2pi = np.array([np.pi / 2, 5 * np.pi / 2])
        else:
            phi_2pi = np.array([])
    for n in range(terms):
        semi_analytic[n] = sT(phi_2pi, xi_2pi, f, theta, bo, ro, n=n)
    return semi_analytic

def compute_disk(f, theta, terms, star):
    tT = np.zeros(terms)
    for n in range(terms):
        tT[n] = tT_numerical(np.array([0,2*np.pi]), (1-f), theta[0], n)
    return tT


##############################################################################
#INTEGRATION BOUNDS
    
def quartic_poly(x, b0, r_p, f, theta):

    x0 = b0*np.sin((theta))
    y0 = b0*np.cos((theta))
    b = 1-f
    A = b**4 - 2*b**2 + 1
    B = 4*x0*(b**2 - 1)
    C = -2*b**4 + 2*b**2*(r_p**2 - x0**2 - y0 + 1) - 2*r_p**2 + 6*x0**2 + 4*y0**2 + 2*y0
    D = 4*x0*(-b**2 + r_p**2 - x0**2 - 2*y0**2 - y0)
    E = (b**4 + 2*b**2*(-r_p**2 + x0**2 + y0) 
         + r_p**4 - 2*r_p**2*x0**2 - 4*r_p**2*y0**2 
         - 2*r_p**2*y0 + x0**4 + 4*x0**2*y0**2 + 2*x0**2*y0 + y0**2)
    
    return (A*x**4 + B*x**3 + C*x**2 + D*x + E)

def quartic_poly_2(x, b0, r_p, f, theta):
    x0 = b0*np.sin(theta)
    y0 = b0*np.cos(theta)
    A, B, C, D, E = quartic_coeffs(1-f,x0,y0,r_p)
    return (A*x**4 + B*x**3 + C*x**2 + D*x + E)

def quartic_coeffs(b, xo, yo, ro):
    A = (1 - b ** 2) ** 2
    B = -4 * xo * (1 - b ** 2)
    C = -2 * (b ** 4 + ro ** 2 - 3 * xo ** 2 - yo ** 2 - b ** 2 * (1 + ro ** 2 - xo ** 2 + yo ** 2))
    D = -4 * xo * (b ** 2 - ro ** 2 + xo ** 2 + yo ** 2)
    E = b ** 4 - 2 * b ** 2 * (ro ** 2 - xo ** 2 + yo ** 2) + (ro ** 2 - xo ** 2 - yo ** 2) ** 2
    return np.array([A, B, C, D, E])



def find_intersections(b0, r_p, f, theta):
    """ 
    b0: impact parameter
    r_p: planet radius
    f: oblateness coefficient
    theta: spin-orbit obliquity *in degrees*
    """
    
    x0 = b0*np.sin(theta)
    y0 = b0*np.cos(theta)
    coeff = quartic_coeffs(1-f,x0,y0,r_p)
    return np.roots(coeff), np.polyval(np.abs(coeff),np.abs(np.roots(coeff)))

def circle_err(x, y, b0, rp, theta):
    return np.abs((y-b0*np.cos(theta))**2-(rp**2 - (x-b0*np.sin(theta))**2))

def compute_xi(b0, r_p, f, theta):
    """ 
    b0: impact parameter
    r_p: planet radius
    f: oblateness coefficient
    theta: spin-orbit obliquity *in degrees*
    """
    
    roots, err = find_intersections(b0, r_p, f, theta)
    roots_real = roots[np.isclose(roots,np.real(roots),atol=0.0001)] #discard imaginary roots
    
    # Mark the points of intersection if they exist
    angles = []
    if len(roots)>0:
        #Xi
        for x_root in roots_real:
            #x value of intersection point
            x_root = np.real(x_root)
            
            #find angles of intersection on ellipse of a=1 and b=(1-f)
            lam_pos = np.arctan2((1-f)*np.sqrt(1-x_root**2),x_root)
            lam_neg = np.arctan2(-(1-f)*np.sqrt(1-x_root**2),x_root)

            r = (1-f)/np.sqrt(((1-f)*np.cos(lam_pos))**2 + np.sin(lam_pos)**2)            
            
            x_int = r*np.cos(lam_pos)
            y_int = r*np.sin(lam_pos)
            
            xi = np.arctan2(np.sqrt(1-x_root**2),x_root)
            xi_neg = np.arctan2(-np.sqrt(1-x_root**2),x_root)     
            x_int_neg = r*np.cos(lam_neg)
            y_int_neg = r*np.sin(lam_neg)
            if circle_err(x_int, y_int, b0, r_p, theta) > circle_err(x_int_neg, y_int_neg, b0, r_p, theta):
                xi = xi_neg
                
            angles.append(xi)
    return np.array(angles)

def compute_phi(b0, r_p, f, theta):
    roots, err = find_intersections(b0, r_p, f, theta)
    roots_real = roots[np.isclose(roots,np.real(roots),atol=0.0001)] #discard imaginary roots
    
    # Mark the points of intersection if they exist
    angles = []
    if len(roots)>0:
        #phi
        for x_root in roots_real:
            #x value of intersection point
            x_root = np.real(x_root)
            
            def rot(b0, theta):
                #return (0,b0) rotated to the integral reference frame F'
                return (b0*np.sin(theta), b0*np.cos(theta))   
            
            #find angles of intersection on ellipse of a=1 and b=(1-f)
            lam_pos = np.arctan2((1-f)*np.sqrt(1-x_root**2),x_root)
            lam_neg = np.arctan2(-(1-f)*np.sqrt(1-x_root**2),x_root)

            r = (1-f)/np.sqrt(((1-f)*np.cos(lam_pos))**2 + np.sin(lam_pos)**2)            
            
            x_int = r*np.cos(lam_pos)
            y_int = r*np.sin(lam_pos)
            x0, y0 = rot(b0, theta)  
            x_int_neg = r*np.cos(lam_neg)
            y_int_neg = r*np.sin(lam_neg)
            
            phi = theta + np.arctan2(y_int-y0,x_root-x0)
            phi_neg = (theta - np.arctan2(-(y_int_neg-y0),x_root-x0))
            
            if circle_err(x_int, y_int, b0, r_p, theta) > circle_err(x_int_neg, y_int_neg, b0, r_p, theta):
                phi = phi_neg
                
            angles.append(phi)
    return np.array(angles)




###############################################################################
#PLOTTING
    
def draw_oblate(b0, rp, f, theta):
    # Set up the figure
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(min(-1.01, -rp - 0.01), max(1.01, rp + 0.01));
    ax.set_ylim(-1.01, max(1.01, b0 + rp + 0.01));
    ax.set_aspect(1);

    # Draw the two bodies
    occulted = Ellipse((0, 0), 1.0*2,(1-f)*2,np.degrees(theta), fill=False, color='k')
    ax.add_artist(occulted)
    occultor = Circle((0, b0), rp, fill=False, color='r')
    ax.add_artist(occultor)
    ax.plot(0, 0, 'ko')
    ax.plot(0, b0, 'ro')
    
    roots, err = find_intersections(b0, rp, f, theta)
    roots_real = roots[np.isclose(roots,np.real(roots),atol=0.0001)] #discard imaginary roots
    
    # Mark the points of intersection if they exist
    if len(roots)>0:
        #Lambda
        for x_root in roots_real:
            x_root = np.real(x_root)
            lam_pos = np.arctan2((1-f)*np.sqrt(1-x_root**2),x_root)
            lam_neg = np.arctan2(-(1-f)*np.sqrt(1-x_root**2),x_root)

            r = (1-f)/np.sqrt(((1-f)*np.cos(lam_pos))**2 + np.sin(lam_pos)**2)            
            
            x = r*np.cos(lam_pos)
            y = r*np.sin(lam_pos)
            
            x_neg = r*np.cos(lam_neg)
            y_neg = r*np.sin(lam_neg)
            special = False
            if circle_err(x, y, b0, rp, theta) > circle_err(x_neg, y_neg, b0, rp, theta):
                x = x_neg
                y = y_neg
            elif np.isclose(circle_err(x, y, b0, rp, theta), circle_err(x_neg, y_neg, b0, rp, theta),atol=1e-8):
                #90 degree case, or too close to tell
                special = True
            
            def rot(x,y, theta):
                #return x,y rotated back into the standard reference frame F
                return (x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta))
            def rot_Fprime(b0, theta):
                #return (0,b0) rotated to the integral reference frame F'
                return (x*np.cos(theta) + y*np.sin(theta), -x*np.sin(theta) + y*np.cos(theta))   
            

            plt.plot([0, rot(x, y, theta)[0]], [0, rot(x, y, theta)[1]], 'k-', alpha=0.3)
            plt.plot(rot(x, y, theta)[0], rot(x, y, theta)[1], 'ko', ms=5)
            plt.plot([0,rot(x, y, theta)[0]],[b0, rot(x, y, theta)[1]], 'r-', alpha=0.3)
            if special:
                plt.plot([0, rot(x_neg, y_neg, theta)[0]], [0, rot(x_neg, y_neg, theta)[1]], 'k-', alpha=0.3)
                plt.plot(rot(x_neg, y_neg, theta)[0], rot(x_neg, y_neg, theta)[1], 'ko', ms=5)
                plt.plot([0,rot(x_neg, y_neg, theta)[0]],[b0, rot(x_neg, y_neg, theta)[1]], 'r-', alpha=0.3)
                
                
def draw_oblate_full(b0, rp, f, theta):
    
    """theta in radians"""
    # Set up the figure
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(min(-1.01, -rp - 0.01), max(1.01, rp + 0.01));
    ax.set_ylim(-1.01, max(1.01, b0 + rp + 0.01));
    ax.set_aspect(1);

    # Draw the two bodies
    occulted = Ellipse((0, 0), 1.0*2,(1-f)*2,np.degrees(theta), fill=False, color='k')
    occulted_circ = Circle((0, 0), 1.0, fill=False, color='b',alpha=0.5)
    ax.add_artist(occulted)
    ax.add_artist(occulted_circ)
    occultor = Circle((0, b0), rp, fill=False, color='r')
    ax.add_artist(occultor)
    ax.plot(0, 0, 'ko')
    ax.plot(0, b0, 'ro')
    
    roots, err = find_intersections(b0, rp, f, theta)
    roots_real = roots[np.isclose(roots,np.real(roots),atol=0.0001)] #discard imaginary roots
    # Mark the points of intersection if they exist
    angles = []
    if len(roots)>0:
        #Xi
        for x_root in roots_real:
            x_root = np.real(x_root)
            
            lam_pos = np.arctan2((1-f)*np.sqrt(1-x_root**2),x_root)
            lam_neg = np.arctan2(-(1-f)*np.sqrt(1-x_root**2),x_root)

            r = (1-f)/np.sqrt(((1-f)*np.cos(lam_pos))**2 + np.sin(lam_pos)**2)            
            
            x_int = r*np.cos(lam_pos)
            y_int = r*np.sin(lam_pos)
            
            xi = np.arctan2(np.sqrt(1-x_root**2),x_root)
            xi_neg = np.arctan2(-np.sqrt(1-x_root**2),x_root)
            
            x = np.cos(xi)
            y = np.sin(xi)           

            x_int_neg = r*np.cos(lam_neg)
            y_int_neg = r*np.sin(lam_neg)
            
            special = False
            if circle_err(x_int, y_int, b0, rp, theta) > circle_err(x_int_neg, y_int_neg, b0, rp, theta):
                x_int = x_int_neg
                y_int = y_int_neg
                x = np.cos(xi_neg)
                y = np.sin(xi_neg)
            elif np.isclose(circle_err(x_int, y_int, b0, rp, theta), circle_err(x_int_neg, y_int_neg, b0, rp, theta),atol=1e-8):
                #90 degree case, or too close to tell
                special = True
            
            
            def rot(x,y, theta):
                #return x,y rotated back into the standard reference frame F
                return (x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta))
            #plot the intersection points and draw a blue line from the origin to the outer radius of the bounding circle
            plt.plot([0, rot(x, y, theta)[0]], [0, rot(x, y, theta)[1]], 'b-', alpha=0.3)
            plt.plot(rot(x, y, theta)[0], rot(x, y, theta)[1], 'ko', ms=5)
            plt.plot(rot(x_int, y_int, theta)[0], rot(x_int, y_int, theta)[1], 'ko', ms=5)
            
            #plot the red line from center of planet to intersectioon points
            plt.plot([0,rot(x_int, y_int, theta)[0]],[b0, rot(x_int, y_int, theta)[1]], 'r-', alpha=0.3)

            xs = [rot(x, y, theta)[0],rot(x, 0, theta)[0]]
            ys = [rot(x, y, theta)[1],rot(x, 0, theta)[1]]
            
            smax = [rot(-1, 0, theta)[0], rot(1, 0, theta)[0]]
            smay = [rot(-1, 0, theta)[1], rot(1, 0, theta)[1]]
            plt.plot(xs, ys, 'k-', alpha=0.4)
            plt.plot(smax, smay, 'k-', alpha=0.4)

            angles.append(xi)
        return np.degrees(angles)