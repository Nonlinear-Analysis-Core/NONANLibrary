import numpy as np
from numpy import matlib

def Div_KL(P,Q):
    """
     dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
     distributions
     P and Q  are automatically normalised to have the sum of one on rows
    have the length of one at each 
    P =  n x nbins
    Q =  1 x nbins or n x nbins(one to one)
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
        P = np.array(P,ndmin=2)
    
    if not isinstance(Q, np.ndarray):
        Q = np.array(Q,ndmin=2)

    if P.shape[1] != Q.shape[1]:
        raise ValueError('the number of columns in P and Q should be the same')

    if not np.isfinite(P).any() or not np.isfinite(Q).any():
        raise ValueError('the inputs contain non-finite values.')

    # normalizing the P and Q
    # if Q has one row.
    if Q.shape[0] == 1:
        Q = np.divide(Q, np.sum(Q))
        P = np.divide(P, matlib.repmat(np.sum(P,axis=1,keepdims=True),1,P.shape[1]))  # repeat the sum of the rows len(rows) times.
        dist = np.sum(np.multiply(P,np.log(np.divide(P,matlib.repmat(Q,P.shape[0],1)))),axis=1) # repeat the values of Q len(col) times.
    elif Q.shape[0] == P.shape[0]:
        Q = np.divide(Q,matlib.repmat(np.sum(Q,axis=1,keepdims=True),1,Q.shape[1]))   
        P = np.divide(P,matlib.repmat(np.sum(P,axis=1,keepdims=True),1,P.shape[1]))     # NOTE: Used to be a 9 not a 1
        dist = np.sum(np.multiply(P,np.log(np.divide(P,Q))),axis=1)

    # resolving the case when P(i)==0
    dist[np.isnan(dist)]=0
    return dist