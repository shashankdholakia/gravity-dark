#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 02:27:41 2020

@author: shashank
"""

from sympy import symbols, Matrix, Rational, floor, sqrt, zeros
from sympy import simplify, factorial, pi, binomial, expand
from sympy.functions.special.tensor_functions import KroneckerDelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

##############################################################################
#SPHERICAL HARMONICS AND GRAVITY DARKENING FUNCTIONS
x, y = symbols('x y', real=True)
omega, G, M, R_eq, beta, alpha = symbols("omega G M R_eq beta alpha",real=True,positive=True)

def binomial_exp(x,k,n):
    assert k==int(k)
    return binomial(n,k)*x**(k)

def poly_basis(n, x, y):
    """Return the n^th term in the polynomial basis."""
    l = Rational(floor(sqrt(n)))
    m = Rational(n - l * l - l)
    mu = Rational(l - m)
    nu = Rational(l + m)
    if nu % 2 == 0:
        i = Rational(mu, 2)
        j = Rational(nu, 2)
        k = Rational(0)
    else:
        i = Rational(mu - 1, 2)
        j = Rational(nu - 1, 2)
        k = Rational(1)
    return x ** i * y ** j * sqrt(1 - x ** 2 - y ** 2) ** k

def Coefficient(expression, term):
    """Return the coefficient multiplying `term` in `expression`."""
    # Get the coefficient
    coeff = expression.coeff(term)
    if term==1:
        coeff = expression.subs(sqrt(1 - x ** 2 - y ** 2), 0).subs(x, 0).subs(y, 0)
    # Set any non-constants in this coefficient to zero. If the coefficient
    # is not a constant, this is not the term we are interested in!
    coeff = coeff.subs(sqrt(1 - x ** 2 - y ** 2), 0).subs(x, 0).subs(y, 0)
    return coeff

def SA(l, m):
    """A spherical harmonic normalization constant."""
    return sqrt(
        (2 - KroneckerDelta(m, 0))
        * (2 * l + 1)
        * factorial(l - m)
        / (4 * pi * factorial(l + m))
    )


def SB(l, m, j, k):
    """Another spherical harmonic normalization constant."""
    try:
        ratio = factorial(Rational(l + m + k - 1, 2)) / factorial(
            Rational(-l + m + k - 1, 2)
        )
    except ValueError:
        ratio = 0
    res = (
        2 ** l
        * Rational(
            factorial(m),
            (factorial(j) * factorial(k) * factorial(m - j) * factorial(l - m - k)),
        )
        * ratio
    )
    return simplify(res)


def SC(p, q, k):
    """Return the binomial theorem coefficient `C`."""
    res = factorial(Rational(k, 2)) / (
        factorial(Rational(q, 2))
        * factorial(Rational(k - p, 2))
        * factorial(Rational(p - q, 2))
    )
    return simplify(res)


def Y(l, m, x, y):
    """Return the spherical harmonic of degree `l` and order `m`."""
    res = 0
    z = sqrt(1 - x ** 2 - y ** 2)
    if m >= 0:
        for j in range(0, m + 1, 2):
            for k in range(0, l - m + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        res += (
                            (-1) ** ((j + p) // 2)
                            * SA(l, m)
                            * SB(l, m, j, k)
                            * SC(p, q, k)
                            * x ** (m - j + p - q)
                            * y ** (j + q)
                        )
            for k in range(1, l - m + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        res += (
                            (-1) ** ((j + p) // 2)
                            * SA(l, m)
                            * SB(l, m, j, k)
                            * SC(p, q, k - 1)
                            * x ** (m - j + p - q)
                            * y ** (j + q)
                            * z
                        )
    else:
        for j in range(1, abs(m) + 1, 2):
            for k in range(0, l - abs(m) + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        res += (
                            (-1) ** ((j + p - 1) // 2)
                            * SA(l, abs(m))
                            * SB(l, abs(m), j, k)
                            * SC(p, q, k)
                            * x ** (abs(m) - j + p - q)
                            * y ** (j + q)
                        )
            for k in range(1, l - abs(m) + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        res += (
                            (-1) ** ((j + p - 1) // 2)
                            * SA(l, abs(m))
                            * SB(l, abs(m), j, k)
                            * SC(p, q, k - 1)
                            * x ** (abs(m) - j + p - q)
                            * y ** (j + q)
                            * z
                        )

    return res


def p_Y(l, m, lmax):
    """Return the polynomial basis representation of the spherical harmonic `Y_{lm}`."""
    ylm = Y(l, m, x, y)
    res = [ylm.subs(sqrt(1 - x ** 2 - y ** 2), 0).subs(x, 0).subs(y, 0)]
    for n in range(1, (lmax + 1) ** 2):
        res.append(Coefficient(ylm, poly_basis(n, x, y)))
    return res


def A1(lmax, norm=2 / sqrt(pi)):
    """Return the change of basis matrix A1. The columns of this matrix are given by `p_Y`."""
    res = zeros((lmax + 1) ** 2, (lmax + 1) ** 2)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            res[n] = p_Y(l, m, lmax)
            n += 1
    return res * norm

def poly_approx(order_approx):
    
    f = 0
    for i in range(order_approx+1):
        f+=binomial_exp(x=(alpha*(x**2+y**2)),k=i,n=2*beta).expand(func=True)
        
    basis = Matrix([poly_basis(n, x, y) for n in range((2*order_approx+1)**2)]).T #need (2*order_approx-2)**2 terms to correctly model
    vec = Matrix([Coefficient(expand(f), term) for term in basis])
    vec = vec.subs([(alpha,(-2*omega**2 + omega**4))])
    return vec

def spherical_approx(order_approx):
    
    """
    Main function to compute spherical harmonic coefficients using a binomial
    expansion of the Von Zeipel Law
    
    Takes an order (of the expansion)
    
    """

    
    f = 0
    for i in range(order_approx+1):
        f+=binomial_exp(x=(alpha*(x**2+y**2)),k=i,n=2*beta).expand(func=True)
        
    basis = Matrix([poly_basis(n, x, y) for n in range((2*order_approx+1)**2)]).T #need (2*order_approx-2)**2 terms to correctly model
    vec = Matrix([Coefficient(expand(f), term) for term in basis])
    
    change_basis_sph = Matrix(A1(2*(order_approx))).inv()
    ycoeffs = simplify(change_basis_sph * vec)
    ycoeffs = ycoeffs.subs([(alpha,(-2*omega**2 + omega**4))])
    return omega, beta, ycoeffs


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
#GREENS THEOREM
    
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
        
        G = [lambda x, y: (1-f) * x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0]
    
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

def primitive(x, y, dx, dy, theta1, theta2, b, n=0):
    """A general primitive integral computed numerically."""

    def func(theta):
        f = 1-b
        Gx, Gy = G(n, f)
        return Gx(x(theta), y(theta)) * dx(theta) + Gy(x(theta), y(theta)) * dy(theta)

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
        res += primitive(x, y, dx, dy, xi1, xi2, b,  n)
    return res

def pT(phi, bo, ro, b, n=0):
    """Compute the pT integral numerically from its integral definition."""
    res = 0
    for phi1, phi2 in phi.reshape(-1, 2):
        x = lambda phi: ro * np.cos(phi)
        y = lambda phi: bo + ro * np.sin(phi)
        dx = lambda phi: -ro * np.sin(phi)
        dy = lambda phi: ro * np.cos(phi)
        res += primitive(x, y, dx, dy, phi1, phi2, b, n)
    return res

def sT(phi, xi, f, theta, b0, rp, n=0):
    """The solution vector for occultations, computed via Green's theorem."""
    b=1-f
    return pT(phi, b0, rp, b, n) + tT_numerical(xi, b, theta, n)

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
    term = g(x, y/(1-f), n=n)
    
    # And sum it over the integration region
    # Each pixel has area dA = (area of square) / (number of pixels)
    dA = 4 / (x.shape[0] * x.shape[1])
    integral = np.nansum(term[in_star & under_occultor] * dA)
    
    return integral

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
