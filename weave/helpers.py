import numpy as np

def T_star_approx(A, a, F, L):
    if F*L == 0:
        return A * ( 1.0 - np.exp( -2.0 * a * np.sin(1.0 / (2.0*a)) ) )
    k = A / (F*L)
    api = a*np.pi
    numer = 4.0 * api**2 * k
    ninin = 1.0 - 2.0*api*k
    dinin = 4.0*api*a*k
    din1 = np.exp(2.0*a*np.sin( ninin / dinin ) ) - 1.0
    din2 = k * din1 - 1.0
    din3 = 1.0 + 2.0*api*din2
    denom = 2.0*api*k * din3 - 1.0
    return -(F*L) / np.log(1.0 + numer / denom)

def T_star_approx2(A, F, L):
    x = F*L / (2*np.pi*A)
    lamb = np.sqrt(1 - x**2) / x
    beta = np.pi - np.arctan(lamb)
    return -F * L / np.log(1 - np.pi / (beta + lamb))

