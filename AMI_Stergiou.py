import numpy as np
import scipy.sparse as sp
import sys

def AMI_Stergiou(data, L, *argv):
    """
    inputs    - data, column oriented time series
              - L, maximal lag to which AMI will be calculated
              - bins, number of bins to use in the calculation, if empty an
                adaptive formula will be used
    outputs   - tau, first minimum in the AMI vs lag plot
              - v_AMI, vector of AMI values and associated lags

    Remarks
    - This code uses average mutual information to find an appropriate lag
      with which to perform phase space reconstruction. It is based on a
      histogram method of calculating AMI.
    - In the case a value of atu could not be found before L the code will
      automatically re-execute with a higher value of L, and will continue to
      re-execute up to a ceiling value of L.

    Future Work
    - None currently.

    Mar 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Modified code to output a plot and notify the user if a value
                of tau could not be found.
    Sep 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Previously the number of bins was hard coded at 128. This
                created a large amount of error in calculated AMI value and
                vastly decreased the sensitivity of the calculation to changes
                in lag. The number of bins was replaced with an adaptive
                formula well known in statistics. (Scott 1979
              - The previous plot output was removed.
    Oct 2017 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Added print commands to display progress.
    May 2019 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - In cases where L was not high enough to find a minimun the
                code would reexecute with a higher L, and the binned data.
                This second part is incorrect and was corrected by using
                data2.
              - The reexecution part did not have the correct input
                parameters.
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
    N = len(data) 

    eps = np.finfo(float).eps # smallest floating point value

    data = np.array(data)

    if len(argv) == 0:
      bins = np.ceil((np.max(data) - np.min(data))/(3.49 * np.nanstd(data * N**(-1/3), axis=0)))
    else:
      bins = argv[0]
    
    bins = int(bins) 

    data = data - min(data) # make all data points positive
    data2 = np.floor(data/(np.max(data)/(bins - eps)))

    v = np.zeros((L,1)) # preallocate the vector
    overlap = N - L
    increment = 1/overlap

    data2 = np.array(data2,dtype=int) # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

    pA = sp.csr_matrix((np.full(overlap,increment),(data2[0:overlap],np.ones(overlap,dtype=int)))).toarray()[:,1]
    

    v = np.zeros((2, L))

    for lag in range(L): # used to be from 0:L-1 (BS)
      v[0,lag]=lag   
      
      pB = sp.csr_matrix((np.full(overlap,increment),(data2[lag:overlap+lag],np.ones(overlap,dtype=int)))).toarray()[:,1]
      # find joint probability p(A,B)=p(x(t),x(t+time_lag))
      pAB = sp.csr_matrix((np.full(overlap,increment), (data2[0:overlap], data2[lag:overlap+lag])))
      
      (A, B) = np.nonzero(pAB)
      AB = pAB.data

      v[1,lag] = np.sum(np.multiply(AB,np.log2(np.divide(AB,np.multiply(pA[A],pB[B]))))) # Average Mutual Information
        
    tau = np.array(np.full((L,2),-1,dtype=float))

    j = 0
    for i in range(v.shape[1]):                       # Finds first minimum
      if v[1,i-1]>=v[1,i] and v[1,i]<=v[1,i+1]: 
        ami = v[1,i]
        tau[j,:] = np.array([i,ami])
        j+=1

    tau = tau[:j]   # only include filled in data.

    initial_AMI = v[1,0]
    
    for i in range(v.shape[1]):                       # Finds first AMI value that is 20% initial AMI
      if v[1,i] < (0.2*initial_AMI):
        tau[0,1] = i
        break

    v_AMI=v

    if len(tau) == 0:      
      if L*1.5 > len(data/3):
        tau[0] = 9999
      else:
        print('Max lag needed to be increased for AMI_Stergiou\n')
        (tau, v_AMI) = AMI_Stergiou(data,np.floor(L*1.5))    #Recursive call

    return (tau, v_AMI)
