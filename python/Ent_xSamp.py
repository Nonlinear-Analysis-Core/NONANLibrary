import numpy as np

def Ent_xSamp(x,y,m,R,norm):
    """
    xSE = Ent_xSamp20180320(x,y,m,R,norm)
    Inputs - x, first data series
           - y, second data series
           - m, vector length for matching (usually 2 or 3)
           - R, R tolerance to find matches (as a proportion of the average 
                of the SDs of the data sets, usually between 0.15 and 0.25)
           - norm, normalization to perform
             - 1 = max rescale/unit interval (data ranges in value from 0 - 1
               ) Most commonly used for RQA.
             - 2 = mean/Zscore (used when data is more variable or has 
               outliers) normalized data has SD = 1. This is best for cross 
               sample entropy.
             - Set to any value other than 1 or 2 to not normalize/rescale 
               the data
    Remarks
    - Function to calculate cross sample entropy for 2 data series using the
      method described by Richman and Moorman (2000).
    Sep 2015 - Created by John McCamley, unonbcf@unomaha.edu
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
    
    # Make sure to have items as numpy arrays
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Check both sets of data are the same length
    
    if x.shape[0] != y.shape[0]: raise ValueError('The data series provided are not the same length')

    N = x.shape[0]
    # normalize the data ensure data fits in the same "space"
    if norm == 1: #normalize data to have a range 0 - 1
        xn = (x - np.min(x))/(np.max(x) - np.min(x))
        yn = (y - np.min(y))/(np.max(y) - np.min(y))
        r = R * ((np.std(xn)+np.std(yn))/2)
    elif norm == 2: # normalize data to have a SD = 1, and mean = 0
        xn = (x - np.mean(x))/np.std(x)
        yn = (y - np.mean(y))/np.std(y)
        r = R
    else: print('These data will not be normalized')

    dij = np.zeros((N-m,m+1))
    dj =  np.zeros((N-m,1))
    dj1 = np.zeros((N-m,1))
    Bm = np.zeros((N-m,1))
    Am = np.zeros((N-m,1))

    for i in range(N-m):
        for k in range(m+1):
            dij[:,k] = np.abs(xn[k:N-m+k]-yn[i+k]) 
        dj = np.max(dij[:,0:m],axis=1)
        dj1 = np.max(dij,axis=1)
        d = np.where(dj<=r) 
        d1 = np.where(dj1<=r)
        nm = d[0].shape[0]
        Bm[i] = nm/(N-m)
        nm1 = d1[0].shape[0]
        Am[i] = nm1/(N-m)

    Bmr = np.sum(Bm)/(N-m)
    Amr = np.sum(Am)/(N-m)

    xSE = np.negative(np.log(Amr/Bmr))
    return xSE

