# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2016

@author: Mathieu FONTAINE, mail: mathieu.fontaine@inria.fr

"""
from __future__ import division
import numpy as np

"""alpha stable functions"""


def random_stable(alpha, beta, mu, sigma, shape, seed=None):
    """
       Input
       -----
       alpha: 0 < alpha <=2
           exponential characteristic coefficient
       beta: -1 <= beta <= 1
           skewness parameter
       mu: real
           the mean
       sigma: positive real
              scale parameter
       shape: as you want :) (give a tuple)
              size and number of sampling

       Returns
       -------
       S: shape
           give a sampling of an S(alpha, beta, mu, sigma) variable

       """
    if seed is None:
        W = np.random.exponential(1, shape)
        U = np.random.uniform(-np.pi / 2., np.pi / 2., shape)

        c = -beta * np.tan(np.pi * alpha / 2.)
        if beta != 1:
            ksi = 1 / alpha * np.arctan(-c)
        else:
            ksi = np.pi / 2.

        res = ((1. + c ** 2) ** (1. / 2. / alpha)) * \
            np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) * \
            ((np.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)
    else:
        _random = np.random.RandomState(seed)
        W = _random.exponential(1, shape)
        U = _random.uniform(-np.pi / 2., np.pi / 2., shape)

        c = -beta * np.tan(np.pi * alpha / 2.)
        if beta != 1:
            ksi = 1 / alpha * np.arctan(-c)
        else:
            ksi = np.pi / 2.

        res = ((1. + c ** 2) ** (1. / 2. / alpha)) * \
            np.sin(alpha * (U + ksi)) / ((np.cos(U)) ** (1. / alpha)) * \
            ((np.cos(U - alpha * (U + ksi))) / W) ** ((1. - alpha) / alpha)

    return res * sigma + mu


def random_complex_isotropic(alpha=1.2, sigma=1, shape=()):
    """
        Input
        -----
        alpha: 1e-20 < alpha <=1.9999
            exponential characteristic coefficient
        sigma: positive real
               scale parameter
        shape: as you want :) (give a tuple)
               size and number of sampling

        Returns
        -------
        S: shape
            give a sampling of an isotropic complex variable SalphaS_c(sigma)

        """
    beta = 1
    # scale parameter
    sigma_imp = 2 * np.cos(np.pi * alpha / 4.) ** float(2. / alpha)
    # impulse variable
    imp = random_stable(alpha / 2., beta, 0, sigma_imp, shape)

    # Complex Gaussian variable
    # real part
    sr = np.random.randn(*shape) * np.sqrt(np.abs(imp)) * sigma * np.sqrt(0.5)
    # imaginary part
    si = np.random.randn(*shape) * np.sqrt(np.abs(imp)) * sigma * np.sqrt(0.5)
    S = sr + 1j * si  # that's our sample of isotropic stable random variable

    return S, imp


def oracle_wiener(sigma_sp, sigma_no, alpha_sp, alpha_no, phi_sp, phi_no):
    eps = 1e-15
    w_sp = phi_sp*sigma_sp**(2) / (
        phi_no*sigma_no**(2) + phi_sp*sigma_sp**(2) + eps
    )
    w_no = phi_no*sigma_no**(2) / (
        phi_no*sigma_no**(2) + phi_sp*sigma_sp**(2) + eps
    )
    return w_sp.squeeze(), w_no.squeeze()


def estimation_wiener(
    sigma_sp, sigma_no, p_sp, p_no, beta_0, beta_1, beta_2, beta_3
):
    eps = 1e-15
    w_estimation_sp = beta_0*sigma_sp**(p_sp) / (
        beta_2*sigma_no**(p_no) + beta_1*sigma_sp**(p_sp) + eps
    )
    w_estimation_no = 1-w_estimation_sp
    return w_estimation_sp.squeeze(), w_estimation_no.squeeze()
