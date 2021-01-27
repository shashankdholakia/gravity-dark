#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:44:42 2021

@author: shashank
"""
from sympy import symbols, Matrix, Rational, floor, sqrt, zeros
from sympy import simplify, factorial, pi, binomial, expand
from sympy.functions.special.tensor_functions import KroneckerDelta


##############################################################################
#SPHERICAL HARMONICS AND GRAVITY DARKENING FUNCTIONS
    
x, y = symbols('x y', real=True)
omega, beta, alpha = symbols("omega beta alpha",real=True,positive=True)

def spherical_approx(order_approx):
    
    """
    Old function to compute spherical harmonic coefficients using a binomial
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