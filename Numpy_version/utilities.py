import numpy as np
from scipy.special import binom
# ---------------------------------------------------------------------------------------------------------
# -----Activation Functions--------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Sigmoid activation function
# ---------------------------------------------------------------------------------------------------------


def sigmoid(x, n):
    '''
    Sigmoid activation function and its first three derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    x = np.maximum(-10, x)
    x = np.minimum(10, x)
    temp_sig = np.exp(-np.round(x,5), dtype="float128")
    temp_sig = 1 / (temp_sig + 1)
    if n == 0:
        return temp_sig
    elif n == 1:
        return temp_sig * (1 - temp_sig)
    elif n == 2:
        return temp_sig * (1 - temp_sig) * (1 - 2 * temp_sig)
    elif n > 2:
        derivative = 0
        for k in range(n+1):
            for j in range(k+1):
                derivative += (-1) ** j * (j+1)**n * binom(k, j) * temp_sig ** (k+1)
        return derivative
    else:
        raise Exception(
            f'Parameter n should be non-negative integer not higher than 4, but n = {n}')
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Linear activation function
# ---------------------------------------------------------------------------------------------------------


def linear(x, n):
    '''
    Linear activation function and its derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    if n == 0:
        return x
    elif n == 1:
        return 1
    elif n > 1:
        return 0
    else:
        raise Exception(
            f'Parameter n should be non-negative integer, but n = {n}')
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# ReLu activation function
# ---------------------------------------------------------------------------------------------------------


def ReLu(x, n):
    '''
    Relu function f(x)=max(0, x) and its derivatives.
    @params:
    1. x - float, the argument,
    2. n - non-negative integer, n-th derivative
    @returns: float
    '''
    if n == 0:
        return np.max(0, x)
    elif n == 1:
        return 0 if x < 0 else 1
    elif n > 1:
        return 0
    else:
        raise Exception(
            f'Parameter n should be non-negative integer, but n = {n}')
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# Kronecker delta function
# ---------------------------------------------------------------------------------------------------------
def kronecker_delta(i, j):
    '''
    Kronecker delta function.
    @params:
    i, j - non-negative integers
    @returns: boolean
    '''
    if isinstance(i, int) and isinstance(j, int) and i >= 0 and j >= 0:
        return i == j
    else:
        raise Exception(
            f'The arguments should be non-negative integers, but i = {i} and j = {j}.')
# ---------------------------------------------------------------------------------------------------------