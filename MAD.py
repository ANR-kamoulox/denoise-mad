# coding=utf-8
"""
Created on Fri, 25th Mar 2016

@author: Mathieu Fontaine, mail: mathieu.fontaine@inria.fr
         Antoine Liutkus,  mail: antoine.liutkus@inria.fr


MAD (MultiAlpha Denoising) algorithm

Robust denoising based on alpha-stable theory (ONLY monochannel case).
"""

import numpy as np
import mpmath
from basics import stft, smooth, alpha_stable_module
import argparse
import soundfile as sf


def separate(
    sig,
    rate,
    nb_it=4,
    alpha_s=1.2,
    alpha_no=1.89,
    deltaTs=5,
    deltaTno=40
):
    """
        Input
        -----
        audio: nd.array
            input audio

        rate: int
            input sample rate

        nb_it: int
            number of iteration of MAD

        alpha_s: 0 < double <= 2
            characteristic exponent (impulsiveness) of the target source

        alpha_no: 0 < double <= 2
            characteristic exponent (impulsiveness) of the noise

        deltaTs: int
            number of time frame for average horizon (speech)

        deltaTno: int
            number of time frame for average horizon (speech)
                  (for a more stationnary noise ==> deltaTno >> deltaTs)


        Returns
        -------
        denoising recording in outputDir

    """

    """Load Files and Initialization """

    Ce = float(mpmath.euler)  # Euler constant
    eps = 1e-3  # for scale parameter estimation & Wiener mask

    nfft = 2048  # number of window
    overlap = 0.60  # 0.xx = xx% of overlap between windows
    hop = float(nfft) * (1.0 - overlap)

    # stft
    X = stft.stft(sig[:, 0], nfft, hop, real=True).astype(np.complex64)
    # scale parameter of the target source
    sigma_phi_s = 2*np.cos(np.pi * alpha_s / 4.) ** float(2. / alpha_s)
    # scale parameter of the noise
    sigma_phi_no = 2*np.cos(np.pi * alpha_no / 4.) ** float(2. / alpha_no)
    phi_sp = alpha_stable_module.random_stable(
        alpha_s / 2., 1, 0, sigma_phi_s,
        (10000, 1)
    )  # sampling of the impulse variable of the target

    # sampling of the impulse variable of the noise
    phi_no = alpha_stable_module.random_stable(
        alpha_no / 2., 1, 0, sigma_phi_no,
        (10000, 1)
    )

    # instead of phi_s, use the median in Wiener filtering
    med_phi_s = np.median(phi_sp)
    # instead of phi_no, use the median in Wiener filtering
    med_phi_no = np.median(phi_no)

    # initialization of the target source signal
    signal_s = X[...]
    # initialization of the noise
    signal_no = X[...]

    # # for a fix number of iteration selected by the user
    for it in range(nb_it):

        # estimation of target source scale parameter
        sigma_s_local = np.log(eps+np.abs(signal_s))
        sigma_s_local = smooth.smooth(sigma_s_local.T, deltaTs).T
        sigma_s = np.exp(sigma_s_local - Ce*(1./alpha_s-1.))

        # estimation of noise scale parameter
        sigma_no_local = np.log(eps+np.abs(signal_no))
        sigma_no_local = smooth.smooth(sigma_no_local.T, deltaTno).T
        sigma_no = np.exp(sigma_no_local - Ce*(1./alpha_no-1.))

        #  parameterized Wiener mask
        G_s = ((med_phi_s * sigma_s ** 2) / (
           eps + med_phi_s * sigma_s ** 2 + med_phi_no * sigma_no ** 2))

        # FILTER ACTION !
        signal_s = G_s * X
        signal_no = (1. - G_s) * X

    # Inverse STFT
    target_source = stft.istft(
        signal_s, 1, hop, real=True, shape=sig[:, 0].shape
    ).astype(np.float32)
    return target_source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load audio and perform separation'
    )
    parser.add_argument(
        'input_file',
        help='audio file'
    )

    args = parser.parse_args()
    sig, rate = sf.read(args.input_file)
    print(sig.shape)
    out = separate(sig, rate)
    sf.write("denoised.wav", data=out, samplerate=rate)
    print(out.shape)
