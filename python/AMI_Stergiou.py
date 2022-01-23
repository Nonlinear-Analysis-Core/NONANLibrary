import numpy as np
import scipy.sparse as sp
import sys

def AMI_Stergiou(data, L, to_matlab = False, n_bins = 0):
    """
    inputs    - data, column oriented time series
              - L, maximal lag to which AMI will be calculated
              - bins, number of bins to use in the calculation, if empty an
                adaptive formula will be used
              - to_matlab, an option for MATLAB users of the code, if MATLAB
                datatypes are needed for output, use this to have them
                returned with proper types. Default is false. 
                
                Only use if you have 'matlab.engine' installed in your current 
                Python env.

                Note: this cannot be installed through the usual conda or pip
                commands, search online to view resources to help in installing
                'matlab.engine' for Python.

    outputs   - tau, first minimum in the AMI vs lag plot
              - v_AMI, vector of AMI values and associated lags
    
    inputs    - x, single column array with the same length as y.
              - y, single column array with the same length as x.
    outputs   - ami, the average mutual information between the two arrays

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
    eps = np.finfo(float).eps # smallest floating point value

    if isinstance(L, int):
      N = len(data) 


      data = np.array(data)

      if n_bins == 0:
        bins = np.ceil((np.max(data) - np.min(data))/(3.49 * np.nanstd(data * N**(-1/3), axis=0)))
      else:
        bins = n_bins
      
      bins = int(bins) 

      data = data - min(data) # make all data points positive
      y = np.floor(data/(np.max(data)/(bins - eps)))
      y = np.array(y,dtype=int) # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

      v = np.zeros((L,1)) # preallocate the vector
      overlap = N - L
      increment = 1/overlap

      pA = sp.csr_matrix((np.full(overlap,increment),(y[0:overlap],np.ones(overlap,dtype=int)))).toarray()[:,1]

      v = np.zeros((2, L))

      for lag in range(L): # used to be from 0:L-1 (BS)
        v[0,lag]=lag   
        
        pB = sp.csr_matrix((np.full(overlap,increment),(y[lag:overlap+lag],np.ones(overlap,dtype=int)))).toarray()[:,1]
        # find joint probability p(A,B)=p(x(t),x(t+time_lag))
        pAB = sp.csr_matrix((np.full(overlap,increment), (y[0:overlap], y[lag:overlap+lag])))
        
        (A, B) = np.nonzero(pAB)
        AB = pAB.data

        v[1,lag] = np.sum(np.multiply(AB,np.log2(np.divide(AB,np.multiply(pA[A],pB[B]))))) # Average Mutual Information
          
      tau = np.array(np.full((L,2),-1,dtype=float))

      j = 0
      for i in range(v.shape[1] - 1):                       # Finds first minimum
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

      return (tau, v_AMI)
    elif isinstance(L, np.ndarray) or isinstance(L, list):
      x = data if isinstance(data,np.ndarray) else np.array(data)
      y = L if isinstance(L,np.ndarray) else np.array(L)

      if len(x) != len(y):
        raise ValueError('X and Y must be the same size.')
      
      increment = 1/len(y)
      one = np.ones(len(y),dtype=int)

      bins1 = np.ceil((max(x)-min(x))/(3.49*np.nanstd(x)*len(x)**(-1/3))) # Scott 1979
      bins2 = np.ceil((max(y)-min(y))/(3.49*np.nanstd(y)*len(y)**(-1/3))) # Scott 1979
      x = x - min(x) # make all data points positive
      x = np.floor(x/(max(x)/(bins1 - eps))) # scaling the data
      y = y - min(y) # make all data points positive
      y = np.floor(y/(max(y)/(bins2 - eps))) # scaling the data

      x = np.array(x,dtype=int)
      y = np.array(y,dtype=int)

      increment = np.full(len(y),increment)
      pA = sp.csr_matrix((increment,(x,one))).toarray()[:,1]
      pB = sp.csr_matrix((increment,(y,one))).toarray()[:,1]
      pAB = sp.csr_matrix((increment,(x,y)))
      (A, B) = np.nonzero(pAB)
      AB = pAB.data
      ami = np.sum(np.multiply(AB,np.log2(np.divide(AB,np.multiply(pA[A],pB[B])))))
      
      if to_matlab:
        import matlab
        return ami
      else:
        return ami
    else:
      raise ValueError('Invalid input, read documentation for input options.')

