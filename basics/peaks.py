# -*- coding: utf-8 -*-
"""
Copyright (c) 2015, Antoine Liutkus, Inria
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

import numpy as np
import scipy.signal as signal


def pickpeaks(V, select, display=False):
    """Scale-space peak picking
    ------------------------
    This function looks for peaks in the data using scale-space theory.

    input :
      * V : data, a vector
      * select : either:
          - select >1 : the number of peaks to detect
          - 0<select<1 : the threshold to apply for finding peaks
            the closer to 1, the less peaks, the closer to 0, the more peaks
      * display : whether or not to display a figure for the results.

    outputs :
      * peaks : indices of the peaks
      * criterion : the value of the computed criterion. Same
                    length as V and giving for each point a high value if
                    this point is likely to be a peak

    The algorithm goes as follows:
    1°) set a smoothing horizon, initially 1;
    2°) smooth the data using this horizon
    3°) find local extrema of this smoothed data
    4°) for each of these local extrema, link it to a local extremum found in
        the last iteration. (initially just keep them all) and increment the
        corresponding criterion using current scale. The
        rationale is that a trajectory surviving such smoothing is an important
        peak
    5°) Iterate to step 2°) using a larger horizon.

    At the end, we keep the points with the largest criterion as peaks.
    I don't know if that kind of algorithm has already been published
    somewhere, I coded it myself and it works pretty nice, so.. enjoy !
    If you find it useful, please mention it in your studies =)

    running time should be decent, although intrinsically higher than
    findpeaks. For vectors of length up to, say, 10 000, it should be nice.
    Above, it may be worth it though.
    ---------------------------------------------------------------------
    (c) Antoine Liutkus, Inria, 2014
    ---------------------------------------------------------------------"""


    # data is a vector
    V = V.flatten() - np.min(V)
    n = V.size

    # definition of local variables
    tempBuffer = np.zeros((n,))
    criterion = np.zeros((n,))
    
    if select < 1:
        minDist = n * select /50
    else:
        minDist = n / select / 8

    #horizons = set(np.round(np.linspace(1,n/2,100)))
    horizons = np.unique(np.maximum(1,np.round(np.linspace(0,1,50) * min(n,minDist*4))))
    horizons.sort()
    Vorig = V

    # loop over scales
    for ind, horizon in enumerate(horizons):
        # sooth data
        V = smooth(V, horizon).flatten()
        I = signal.argrelextrema(V, np.greater)[0]
        if len(I)==1:break
        # initialize buffer
        newBuffer = np.zeros(tempBuffer.shape)

        if not ind:
            # if first iteration, keep all local maxima
            newBuffer[I] = Vorig[I]
        else:
            old = np.nonzero(tempBuffer)[0]
            if not old.size:
                continue
            neighbours = np.digitize(I,old)
            
            
            old = np.concatenate((old,[n-1]))
            dl = np.abs(I-old[neighbours-1])
            dr = np.abs(I - old[neighbours])
            dldr = np.concatenate((dl[...,None],dr[...,None]),axis=1)
            neighbours += -1+np.argmin(dldr,axis=1)
            newBuffer[old[neighbours]] = V[old[neighbours]]  * (ind ** 2) + V[I]*ind
            

        # update stuff
        tempBuffer = newBuffer
        criterion = criterion + newBuffer

    # normalize criterion
    criterion = criterion / np.max(criterion)

    # find peaks based on criterion
    if select < 1:
        peaks = sorted(
            np.nonzero(
                criterion > select),
            key=lambda x: criterion[x])
    else:
        order = np.argsort(criterion)[::-1]
        peaks = order[:select]

    if display:
        import pylab as pl
        # display
        pl.figure(1)
        pl.clf()
        pl.plot(Vorig)
        pl.hold(True)
        pl.plot(criterion * max(Vorig), 'r')
        pl.hold(True)

        pl.plot(np.squeeze(np.array(peaks)), np.squeeze(Vorig[peaks]), 'ro')
        pl.grid(True)
        pl.title('Scale-space peak detection')
        #pl.legend(['data', 'computed criterion', 'selected peaks'])
        pl.show()

    return peaks, criterion

# helper smoothing function
def smooth(s, lengthscale):
    """smoothes s vertically"""
    if lengthscale <= 1:
        return s
    lengthscale = 2 * round(float(lengthscale) / 2.0)
    W = np.hamming(min(lengthscale, s.shape[0]))
    W = W / np.sum(W)
    return signal.fftconvolve(s, W, mode='same')

