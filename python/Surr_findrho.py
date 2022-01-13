import numpy as np
import Surr_PseudoPeriodic as pseudo

def Surr_findrho(y,tau,dim):
    """
    rho=Surr_findrho20200626(y,tau,dim)
    inputs  - y, time series
            - tau, time lag for phase space reconstruction
            - dim, embedding dimension for phase space reconstruction
    outputs - a tuple of the shape (rho, out) where rho and out are defined as:
            - rho, noise radius
            - out, an informational array with results of rho and di from the
                   iterative processes
    Remarks
    - This code finds an optimal value of rho for the pseudo periodic
      surrogation algorithm. It maximizes the number of short sequences in
      the surrogate that are identical to the original time series.
    - The method in this code first finds rho at a range of values untill a
      suspected peak is found. A binary search is then performed around the
      peak untill the percent difference decreases below a threshold.
    Future Work
    - The value rho ts to maximize with a pulse with very small values
      around it. It is suspected this may cause issues with some time series
      but it has not been encountered.
    - It's possible the rhoL value of 0.1 may cause issues with pps in
      certain time series.
    - It's possible a time series may have an optimal value of rho above 0.1
      but this has not been encountered.
    References
    - Small, M., Yu, D., & G., H. R. (2001). Surrogaet Test for
      Pseudoperiodic Time Series Data. Physical Revew Letters, 87(18).
      https://doi.org/10.1063/1.1487534
    Version History
    May 2001 - Created by Michael Small
             - It is believed the original version of this code was written
               by Michael Small but the source could not be confirmed.
    Jun 2020 - Modifed by Ben Senderling
             - Removed input handling, all three are necessary. In this
               version the initial search for the bounds of the binary search
               was removed completely. It was noticed the optimal rho is
               frequently ~0.5-0.6. The bounds were replaced with 0.1 and 1 
               for rhoL and rhoH. This offered a 35# speed improvement over 
               the previous version when testing an ankle angle, n=10800, 10 
               times. Added an out variable to use for troubleshooting and
               diagnostics.
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
       contributors may be used to orse or promote products derived from 
       this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR 
    PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    # Find upper bound for binary search

    rhoH=2.
    (_,yi)=pseudo.Surr_PseudoPeriodic(y,tau,dim,rhoH)
    diH=findrho_di(yi,2)

    out = np.array([1,rhoH,diH])

    # Find lower bound for the binary search

    rhoL=0.1
    (_,yi)=pseudo.Surr_PseudoPeriodic(y,tau,dim,rhoL)
    diL=findrho_di(yi,2)
    out = np.vstack((out, np.array([2,rhoL,diL])))

    # Find which bound has more short continuous sequences from the original 
    # time series.

    if diH>diL:
        dmax=diH
    else:
        dmax=diL
    

    # Perform binary search

    precision=0.02
    ind=3

    rho = np.array([])

    while abs(rhoH - rhoL) / rhoL > precision:
        rhoi = (rhoH+rhoL)/2
        (_,yi) = pseudo.Surr_PseudoPeriodic(y,tau,dim,rhoi)
        di=findrho_di(yi,2)
        out = np.vstack((out,np.array([ind,rhoi,di])))
        ind=ind+1
        
        if di>dmax:
            dmax=di
            rho=rhoi
        if diL<diH:
            diL=di
            rhoL=rhoi
        else:
            diH=di
            rhoH=rhoi
    return (rho,out)

def findrho_di(yi,n):
    di=np.diff(np.where(np.diff(yi,axis=0)==1))[0]
    di=np.sum(di>n)
    return di




