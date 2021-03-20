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
from scipy.special import hyp2f1, ellipeinc
from scipy.special import comb

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
    #rotate so that planet is aligned with x axis (0-angle) instead of the y axis. 
    #then wrap the angle from 0-2 pi and sort. 
    #this way, the integration should always be performed counterclockwise around the right region
    xi_2pi = np.sort(np.mod(xi-np.pi/2, 2*np.pi))+np.pi/2 
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

def analytic_sT(f, theta, bo, ro, terms):
    xi = compute_xi(bo, ro, f, theta)
    phi = compute_phi(bo, ro, f, theta)
    #rotate so that planet is aligned with x axis (0-angle) instead of the y axis. 
    #then wrap the angle from 0-2 pi and sort. 
    #this way, the integration should always be performed counterclockwise around the right region
    xi_2pi = np.sort(np.mod(xi-np.pi/2, 2*np.pi))+np.pi/2 
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
            phi_2pi = np.array([0, 0])
    try:
        assert len(xi_2pi) != 0
    except:
        xi_2pi = np.array([0,0])
    solver = Solver(lmax=int(np.sqrt(terms)-1))
    analytic = solver.get_sT(bo, ro, f, theta, phi_2pi[0], phi_2pi[1], xi_2pi[0], xi_2pi[1])
    return analytic
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
# ANALYTIC INTEGRALS
    
class Solver:
    def __init__(self, lmax):

        # Pre-compute case conditions
        self.lmax = lmax
        self.cases = np.zeros((self.lmax + 1) ** 2, dtype=int)
        self.l = np.zeros((self.lmax + 1) ** 2, dtype=int)
        self.m = np.zeros((self.lmax + 1) ** 2, dtype=int)
        for n in range((self.lmax + 1) ** 2):
            self.cases[n] = self.get_case(n)
            self.l[n] = int(np.floor(np.sqrt(n)))
            self.m[n] = n - self.l[n] ** 2 - self.l[n]

    def get_case(self, n):
        l = int(np.floor(np.sqrt(n)))
        m = n - l ** 2 - l
        mu = l - m
        nu = l + m
        if (nu % 2) == 0:
            return 1
        elif (l == 1) and (m == 0):
            return 2
        elif ((nu % 2) == 1) and (mu == 1) and ((l % 2) == 0):
            return 3
        elif ((nu % 2) == 1) and (mu == 1) and ((l % 2) == 1):
            return 4
        else:
            return 5

    def solve(self, f0, fN, A, B, C, N):
        """
        Solve a recursion relation of the form
        
            f(j) = A(j) * f(j - 1) + B(j) * f (j - 2) + C(j)
        
        given a lower boundary condition f(0) and 
        an upper boundary condition f(N).
        
        """
        # Set up the tridiagonal problem
        a = np.empty(N - 1)
        b = np.empty(N - 1)
        c = np.empty(N - 1)
        for i, j in enumerate(range(2, N + 1)):
            a[i] = -A(j)
            b[i] = -B(j)
            c[i] = C(j)

        # Add the boundary conditions
        c[0] -= b[0] * f0
        c[-1] -= fN

        # Construct the tridiagonal matrix
        A = np.diag(a, 0) + np.diag(b[1:], -1) + np.diag(np.ones(N - 2), 1)

        # Solve using a dense solver for stability
        soln = np.linalg.solve(A, c)
        return np.concatenate(([f0], soln, [fN]))

    def get_W(self, k2, phi1, phi2):
        """
        Return the vector of `W` integrals, computed
        recursively given a lower boundary condition
        (analytic in terms of elliptic integrals) and an upper
        boundary condition (computed numerically).

        The term `W_j` is the solution to the integral of
        
            sin(u)^(2 * j) * sqrt(1 - sin(u)^2 / (1 - k^2))

        from u = u1 to u = u2, where 
        
            u = (pi - 2 * phi) / 4

        """
        # Useful quantities
        kc2 = 1 - k2
        u1 = 0.25 * (np.pi - 2 * phi1)
        u2 = 0.25 * (np.pi - 2 * phi2)
        u = np.array([u1, u2])
        sinu = np.sin(u)
        cosu = np.cos(u)
        diff = lambda x: np.diff(x)[0]
        D = (1 - sinu ** 2 / kc2) ** 0.5

        # The two boundary conditions
        # TODO: How should we evaluate fN?
        N = 2 * self.lmax + 4
        f0 = diff(ellipeinc(u, 1 / kc2))
        fN = quad(
            lambda u: np.sin(u) ** (2 * N) * (1 - np.sin(u) ** 2 / kc2) ** 0.5,
            u1,
            u2,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]

        # The recursion coefficients
        A = lambda j: 2 * (j + (j - 1) * kc2) / (2 * j + 1)
        B = lambda j: -(2 * j - 3) / (2 * j + 1) * kc2
        C = lambda j: diff(kc2 * sinu ** (2 * j - 3) * cosu * D ** 3) / (2 * j + 1)

        # Solve the linear system
        return self.solve(f0, fN, A, B, C, N)

    def get_V(self, k2, phi1, phi2):
        """
        Return the vector of `V` integrals, computed
        recursively given a lower boundary condition
        (trivial) and an upper boundary condition 
        (computed from `2F1`). 
        
        The term `V_i` is the solution to the integral of
        
            cos(phi) sin(phi)^i sqrt(1 - w^2 sin(phi))
        
        from phi=phi1 to phi=phi2.
        
        """
        N = 2 * self.lmax + 4
        invw2 = 2 * k2 - 1

        V = np.zeros((N + 1, 2))
        for i, phi in enumerate([phi1, phi2]):

            # Useful quantities
            sinphi = np.sin(phi)
            x = sinphi / invw2

            # The two boundary conditions
            f0 = (2.0 / 3.0) * (1 - (1 - x) ** 1.5) * invw2
            fN = sinphi ** (N + 1) / (N + 1) * hyp2f1(-0.5, N + 1, N + 2, x)

            # The recursion coefficients
            A = lambda j: (2 * j + (2 * j + 1) * x) * invw2 / (2 * j + 3)
            B = lambda j: -2 * (j - 1) * sinphi * invw2 / (2 * j + 3)
            C = lambda j: 0.0

            # Solve the linear system
            V[:, i] = self.solve(f0, fN, A, B, C, N)

        return np.diff(V, axis=1)

    def get_J(self, k2, phi1, phi2):
        """
        Return the `J` matrix of integrals. The term `J_{p,q}`
        is equal to the integral of
        
            cos(phi)^p sin(phi)^q sqrt(1 - w^2 sin(phi))
            
        from phi=phi1 to phi=phi2.
        
        """
        W = self.get_W(k2, phi1, phi2)
        V = self.get_V(k2, phi1, phi2)
        w2 = 1 / (2 * k2 - 1)
        c1 = -2.0 * (1 - w2) ** 0.5

        J = np.zeros((self.lmax + 3, self.lmax + 3))
        for s in range((self.lmax + 3) // 2):
            for q in range(self.lmax + 3):
                fac = 1
                for i in range(s + 1):
                    term = 0
                    amp = 1
                    for j in range(2 * i + q + 1):
                        term += amp * W[j]
                        amp *= -2 * (2 * i + q - j) / (j + 1)
                    J[2 * s, q] += c1 * fac * term
                    J[2 * s + 1, q] += fac * V[2 * i + q]
                    fac *= (i - s) / (i + 1)
        return J

    def get_L(self, phip1, phip2):
        """
        Return the `L` matrix of integrals. The term `L_{i, j}`
        is equal to the integral of
        
            cos(phi)^i sin(phi)^j
            
        from phi=phip1 to phi=phip2, where
        
            phip = phi - theta
            
        (pT integral) or
        
            phip = xi
            
        (tT integral).
        
        """
        # Lower boundary
        L = np.zeros((self.lmax + 3, self.lmax + 3))
        L[0, 0] = phip2 - phip1
        L[1, 0] = np.sin(phip2) - np.sin(phip1)
        L[0, 1] = np.cos(phip1) - np.cos(phip2)
        L[1, 1] = 0.5 * (np.cos(phip1) ** 2 - np.cos(phip2) ** 2)

        # Recursion coeffs
        cp1 = np.cos(phip1)
        cp2 = np.cos(phip2)
        sp1 = np.sin(phip1)
        sp2 = np.sin(phip2)
        A0 = cp1 * sp1
        B0 = cp2 * sp2
        C0 = cp2 * sp2
        D0 = cp1 * sp1

        # Recurse
        for u in range(self.lmax + 3):
            A = A0
            B = B0
            C = C0
            D = D0
            for v in range(2, self.lmax + 3):
                fac = 1.0 / (u + v)
                L[u, v] = fac * (A - B + (v - 1) * L[u, v - 2])
                L[v, u] = fac * (C - D + (v - 1) * L[v - 2, u])
                A *= sp1
                B *= sp2
                C *= cp2
                D *= cp1
            A0 *= cp1
            B0 *= cp2
            C0 *= sp2
            D0 *= sp1

        return L

    def get_pT2(self, r, theta, f, b, phi1, phi2):
        """
        Return the `pT[2]` integral, computed by numerical integration.
        
        """
        bc = b * np.cos(theta)
        bs = b * np.sin(theta)
        p = (1 - f) ** (-2)

        def func(phi):
            z = np.sqrt(
                1
                - (r * np.cos(phi - theta) + bs) ** 2
                - p * (r * np.sin(phi - theta) + bc) ** 2
            )
            z = max(1e-12, min(z, 1 - 1e-8))
            return r * (r + b * np.sin(phi)) * (1 - z ** 3) / (3 * (1 - z ** 2))

        res, _ = quad(func, phi1, phi2, epsabs=1e-12, epsrel=1e-12)
        return res

    def get_sT(self, b, r, f, theta, phi1, phi2, xi1, xi2):
        """
        Return the `sT` solution vector.
        
        TODO: Instabilities occur when
        
            - np.abs(costheta) < 1e-15
            - np.abs(sintheta) < 1e-15
            - np.abs(b) < 1e-3
            - r < 1e-3
            - np.abs(1 - b - r) < 1e-3
        
        """
        # Useful variables
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        tantheta = sintheta / costheta
        invtantheta = 1.0 / tantheta
        gamma = 1 - b ** 2 - r ** 2
        sqrtgamma = np.sqrt(gamma)
        w2 = 2 * b * r / gamma
        k2 = (gamma + 2 * b * r) / (4 * b * r)

        # Compute the S, C matrices
        S = np.zeros((self.lmax + 3, self.lmax + 3))
        C = np.zeros((self.lmax + 3, self.lmax + 3))
        fac0 = 1
        for i in range(self.lmax + 3):
            facs = r * fac0
            facc = fac0
            for j in range(i + 1):
                S[i, self.lmax + 2 + j - i] = facs
                C[self.lmax + 2 + j - i, i] = facc
                fac = b * (i - j) / (r * (j + 1))
                facs *= fac * sintheta
                facc *= fac * costheta
            fac0 *= r

        # Compute the B matrices
        A = np.empty((self.lmax + 3, self.lmax + 3))
        B = np.empty((self.lmax + 3, self.lmax + 3))
        fac0l = 1
        fac0r = 1
        for i in range(self.lmax + 3):
            facl = fac0l
            facr = fac0r
            for j in range(self.lmax + 3):
                A[i, j] = facl
                B[i, j] = facr
                fac = (i - j) / (j + 1)
                facl *= fac * tantheta
                facr *= -fac * invtantheta
            fac0l *= costheta
            fac0r *= -sintheta

        # Compute the first f derivative matrix
        D = -3 * np.array(
            [
                [
                    0.5 * (b ** 2 + r ** 2 + (b + r) * (b - r) * np.cos(2 * theta)),
                    2 * b * r * costheta ** 2,
                    r ** 2 * np.cos(2 * theta),
                ],
                [
                    -2 * b * r * costheta * sintheta,
                    -2 * r ** 2 * costheta * sintheta,
                    0,
                ],
            ]
        )

        # Compute the L and M integrals
        L = self.get_L(phi1 - theta, phi2 - theta)
        Lxi = (1 - f) * self.get_L(xi1, xi2)
        M = S[:-1, 1:] @ L[::-1, ::-1][:-1, :] @ C[:, :-1]

        # Compute the J integrals
        J = self.get_J(k2, phi1, phi2)
        J32 = gamma * sqrtgamma * (J[:-1, :-1] - w2 * J[:-1, 1:])
        J12 = sqrtgamma * J[:-1, :-1]

        # Compute the I integral
        I = np.zeros((self.lmax + 2, self.lmax + 2))
        T = np.zeros((self.lmax + 1, self.lmax))
        for i in range(self.lmax + 1):
            for j in range(min(self.lmax, i + 1)):

                # Taylor series in `f`
                J0 = J32[i - j, j]
                J1 = np.sum(D * (J12[i - j :, j:])[:2, :3])
                T[i, j] = J0 + f * J1

                for k in range(i + 1):
                    imk = i - k
                    for l in range(max(0, j - imk), min(k, j) + 1):
                        I[self.lmax + 1 - imk, self.lmax + 1 - k] += (
                            A[imk, j - l] * T[i, j] * B[k, l]
                        )

        # Compute the K, H integrals
        K = S[:-1, 2:] @ I[:-1, :] @ C[1:, :-1]
        H = S[:-1, 1:] @ I[:, :-1] @ C[2:, :-1]

        # Compute sT on a case-by-case basis
        sT = np.zeros((self.lmax + 1) ** 2)
        for n in range((self.lmax + 1) ** 2):
            l = self.l[n]
            m = self.m[n]
            mu = l - m
            nu = l + m
            if self.cases[n] == 1:
                pT = (1 - f) ** (-nu // 2) * M[(mu + 2) // 2, nu // 2]
                tT = Lxi[mu // 2 + 2, nu // 2]
                sT[n] = pT + tT
            elif self.cases[n] == 2:
                pT = self.get_pT2(r, theta, f, b, phi1, phi2)
                tT = (1 - f) * (xi2 - xi1) / 3
                sT[n] = pT + tT
            elif self.cases[n] == 3:
                sT[n] = -(1 - f) * H[l - 2, 0]
            elif self.cases[n] == 4:
                sT[n] = -H[l - 3, 1]
            else:
                sT[n] = (1 - f) ** ((1 - nu) // 2) * K[(mu - 3) // 2, (nu - 1) // 2]

        return sT.reshape(1, -1)


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
    