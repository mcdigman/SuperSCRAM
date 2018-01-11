#TODO eliminate this file from main code base if nothing uses it
from math import sqrt, cos, pi

from scipy.integrate import quad
#TODO pretty sure theres a builtin for this
def odd_factorial(k):
    f = k
    while k >= 3:
        k -= 2
        f *= k
    return f

def assoc_legendre_p(nu, mu, x):
    """Compute the associated Legendre polynomial with degree nu and order mu.

    This function uses the recursion formula in the degree nu.
    (Abramowitz & Stegun, Section 8.5.)
    """
    if mu < 0 or mu > nu:
        raise ValueError('require 0 <= mu <= nu, but mu=%d and nu=%d' % (nu, mu))
    if abs(x) > 1:
        raise ValueError('require -1 <= x <= 1, but x=%f', x)

    # Compute the initial term in the recursion.
    if mu == 0:
        p_nu = 1.0
    else:
        s = 1
        if mu & 1:
            s = -1
        z = sqrt(1 - x**2)
        p_nu = s * odd_factorial(2*mu - 1) * z**mu

    if mu == nu:
        return p_nu

    # Compute the next term in the recursion.
    p_nu_prev = p_nu
    p_nu = x * (2*mu + 1) * p_nu

    if nu == mu + 1:
        return p_nu

    # Iterate the recursion relation.
    for n in xrange(mu+2, nu+1):
        result = (x*(2*n - 1)*p_nu - (n + mu - 1)*p_nu_prev)/(n - mu)
        p_nu_prev = p_nu
        p_nu = result

    return result


def integrand(phi, x, nu, mu, sqrt1mx2):
    """Compute (x + sqrt(1-x**2)*cos(phi)*1j)**nu * cos(mu*phi), and return
    one of +/- the real or imaginary part, depending on mu."""
    c = sqrt1mx2 * cos(phi)
    z = (x + c*1j)**nu
    k = mu & 3
    if k == 0:
        r = z.real
    elif k == 1:
        r = -z.imag
    elif k == 2:
        r = -z.real
    else:
        r = z.imag
    result = r * cos(mu*phi)
    return result


def assoc_legendre_p2(nu, mu, x):
    """Compute the associated Legendre polynomial with degree nu and order mu.

    This function uses an integral representation of the polynomial.
    (Courant and Hilbert, "Methods of Mathematical Physics", Vol. 1,
    Section VII.3.4, p. 505.)  It is generally much slower than the
    function assoc_legendre_p.

    It returns a tuple containing the computed value and an estimate of
    the absolute error.
    """
    if mu < 0 or mu > nu:
        raise ValueError('require 0 <= mu <= nu, but mu=%d and nu=%d' % (nu, mu))
    if abs(x) > 1:
        raise ValueError('require -1 <= x <= 1, but x=%f', x)
    c = 1.0/pi
    for n in xrange(1,mu+1):
        c *= nu + n
    # Precompute sqrt(1-x**2) and pass it to the integrand function,
    # instead of computing it every time quad calls integrand.
    sqrt1mx2 = sqrt(1-x**2)
    integral, err = quad(integrand, 0, pi, args=(x, nu, mu, sqrt1mx2), epsabs=1e-14, epsrel=1e-12)
    p = c * integral
    return p, c*err
