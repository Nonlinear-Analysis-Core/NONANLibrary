import numpy as np
from numpy import matlib
from Div_KL import Div_KL

def Div_JS(P,Q):
    """
    Jensen-Shannon divergence of two probability distributions
     dist = JSD(P,Q) Kullback-Leibler divergence of two discrete probability
     distributions
     P and Q  are automatically normalised to have the sum of one on rows
    have the length of one at each 
    P =  n x nbins
    Q =  1 x nbins
    dist = n x 1
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha
    
    Redistribution and use in source and binary forms, with or without 
    modification, are permitted provided that the following conditions are 
    met:
    
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright 
       notice, this list of conditions and the following disclaimer in the 
       documentation and/or other materials provided with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its 
       contributors may be used to endorse or promote products derived from 
       this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    if not isinstance(P, np.ndarray):
        P = np.array(P, ndmin=2)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, ndmin=2)

    if P.shape[1] != Q.shape[1]:
        raise ValueError('The number of columns in P and Q should be the same')

    Q = np.divide(Q,np.sum(Q))
    Q = matlib.repmat(Q, P.shape[0], 1)
    P = np.divide(P,matlib.repmat(np.sum(P,axis=1,keepdims=True),1,P.shape[1]))

    M = np.multiply(0.5,np.add(P,Q))

    dist = np.multiply(0.5,Div_KL(P,M)) + 0.5*Div_KL(Q,M)
    return dist