import numpy as np

def Ent_Samp(data, m, r):
  """
  function SE = Ent_Samp20200723(data,m,r)
  SE = Ent_Samp20200723(data,m,R) Returns the sample entropy value.
  inputs - data, single column time seres
          - m, length of vectors to be compared
          - r, radius for accepting matches (as a proportion of the
            standard deviation)

  output - SE, sample entropy
  Remarks
  - This code finds the sample entropy of a data series using the method
    described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
    time-series analysis using approximate entropy and sample entropy."
    Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.
  - m is generally recommendation as 2
  - R is generally recommendation as 0.2
  May 2016 - Modified by John McCamley, unonbcf@unomaha.edu
           - This is a faster version of the previous code.
  May 2019 - Modified by Will Denton
           - Added code to check version number in relation to a server
             and to automatically update the code.
  Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
           - Removed the code that automatically checks for updates and
             keeps a version history.
  Define r as R times the standard deviation
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
  R = r * np.std(data)
  N = len(data)

  data = np.array(data)

  dij = np.zeros((N-m,m+1))
  dj =  np.zeros((N-m,1))
  dj1 = np.zeros((N-m,1))
  Bm = np.zeros((N-m,1))
  Am = np.zeros((N-m,1))

  for i in range(N-m):
    for k in range(m+1):
      dij[:,k] = np.abs(data[k:N-m+k]-data[i+k]) 
    dj = np.max(dij[:,0:m],axis=1)
    dj1 = np.max(dij,axis=1)
    d = np.where(dj <= R) 
    d1 = np.where(dj1 <= R)
    nm = d[0].shape[0]-1 # subtract the self match
    Bm[i] = nm/(N-m)
    nm1 = d1[0].shape[0]-1 # subtract the self match
    Am[i] = nm1/(N-m)
  
  Bmr = np.sum(Bm)/(N-m)
  Amr = np.sum(Am)/(N-m)

  return -np.log(Amr/Bmr)
