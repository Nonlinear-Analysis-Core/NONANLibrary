import numpy as np

def Ent_MS_Plus(x, tau, m, r):
    """
    (RCMSE, CMSE, MSE, MSFE) = RCMS_Ent( x, tau, m, r )
    inputs - x, single column time seres
           - tau, greatest scale factor
           - m, length of vectors to be compared
           - R, radius for accepting matches (as a proportion of the
                standard deviation)
    output - RCMSE, Refined Composite Multiscale Entropy
           - CMSE, Composite Multiscale Entropy
           - MSE, Multiscale Entropy
           - MSFE, Multiscale Fuzzy Entropy
           - GMSE, Generalized Multiscale Entropy
    Remarks
    - This code finds the Refined Composite Multiscale Sample Entropy,
      Composite Multiscale Entropy, Multiscale Entropy, Multiscale Fuzzy
      Entropy and Generalized Multiscale Entropy of a data series using the
      methods described by - Wu, Shuen-De, et al. 2014. "Analysis of complex
      time series using refined composite multiscale entropy." Physics
      Letters A. 378, 1369-1374.
    - Each of these methods calculates entropy at different scales. These
      scales range from 1 to tau in increments of 1.
    - The Complexity Index (CI) is not calculated by this code. Because the scales
      are incremented by 1 the C is the summation of all the elements in each
      array. For example the CI of MSE would be sum(MSE).
    20170828 Created by Will Denton, bmchnonan@unomaha.edu
    20201001 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Modifed to calculate all scales in this single code instead of
               needing to be in an external for loop.
    """

    R = r*np.std(x)
    N = len(x)

    GMSE = np.zeros(tau, dtype=object)
    MSE = np.zeros(tau, dtype=object)
    MSFE = np.zeros(tau, dtype=object)
    CMSE = np.zeros(tau, dtype=object)
    RCMSE = np.zeros(tau, dtype=object)

    for i in range(tau):
        # Coarse-graining for GMSE
        Ndivi = int(N/i) # defining this now because we use it a lot later on.
        o2 = np.zeros((i, Ndivi))
        for j in range(Ndivi):
            for k in range(i):
                try:
                    # NOTE: May have a one off issue.
                    o2[k,j] = np.var(x[(j-1)*i+k:j*i+k])
                except:
                    o2[k,j] = np.nan
        GMSE[i] = Samp_Ent(o2[0,:],m,r)
        
        # Coarse-graining for MSE and derivatives
        y_tau_kj = np.zeros((i,Ndivi))
        for j in range(Ndivi):
            for k in range(i):
                try:
                    y_tau_kj[k,j] = 1/i*np.sum(x[(j-1)*i+k:j*i+k])
                except:
                    y_tau_kj[k,j] = np.nan
        
        # Multiscale Entropy (MSE)
        MSE[i] = Samp_Ent(y_tau_kj[0, not np.isnan(y_tau_kj[0,:])],m,R)
        
        #Multiscale Fuzzy Entropy (MFE)
        MSFE[i] = Fuzzy_Ent(y_tau_kj[0, not np.isnan(y_tau_kj[0,:])],m,R,2)
        
        # Composite Multiscale Entropy (CMSE)
        CMSE[i] = 0

        nm = np.zeros(i)
        nm1 = np.zeros(i)
        for k in range(i):
            _, nm[k], nm1[k] = Samp_Ent(y_tau_kj[k, not np.isnan(y_tau_kj[k,:])],m,R)
            CMSE[i] = CMSE[i]+1/i*-np.log(nm1[k]/nm[k])
        
        # Refined Composite Multiscale Entropy (RCMSE)
        n_m1_ktau = 1/i*np.sum(nm1)
        n_m_ktau = 1/i*np.sum(nm)
        RCMSE[i] = -np.log(n_m1_ktau/n_m_ktau)

    return RCMSE, CMSE, MSE, MSFE, GMSE

def Samp_Ent(data, m, r):
    """
    [SE,sum_nm,sum_nm1] = Samp_Ent(data,m,r)
    This is a faster version of the previous code - Samp_En.m

    inputs     - data, single column time seres
               - m, length of vectors to be compared
               - R, radius for accepting matches (as a proportion of the
                    standard deviation)

    output     - SE, sample entropy
               - sum_nm, total number of matches for vector length m
               - sum_nm1, total number of matches for vector length m+1
    
    Remarks
    This code finds the sample entropy of a data series using the method
    described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
    time-series analysis using approximate entropy and sample entropy."
    Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.


    J McCamley May, 2016
    W Denton August, 2017 (Made count total number of matches for each vector length, necessary for CMSE and RCMSE)
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
        nm = d[0].shape[0]
        sum_nm = sum_nm + nm
        Bm[i] = nm/(N-m)
        nm1 = d1[0].shape[0]
        sum_nm1 = sum_nm1 + nm1
        Am[i] = nm1/(N-m)
    
    Bmr = np.sum(Bm)/(N-m)
    Amr = np.sum(Am)/(N-m)

    return (-np.log(Amr/Bmr), sum_nm, sum_nm1)

def Fuzzy_Ent(series, dim, r, n):
    """ 
    Function which computes the Fuzzy Entropy (FuzzyEn) of a time series. The
    algorithm presented by Chen et al. at "Charactirization of surface EMG
    signal based on fuzzy entropy" (DOI: 10.1109/TNSRE.2007.897025) has been
    followed.

    INPUT:
            series: the time series.
            dim: the embedding dimesion employed in the SampEn algorithm.
            r: the width of the fuzzy exponential function.
            n: the step of the fuzzy exponential function.

    OUTPUT:
            FuzzyEn: the FuzzyEn value.

    PROJECT: Research Master in signal theory and bioengineering - University of Valladolid

    DATE: 11/10/2014

    VERSION: 1

    AUTHOR: Jess Monge lvarez
    """
    # Checking the input parameters:
    # Processing:
    # Normalization of the input time series:
    # series = (series-mean(series))/std(series);
    N = len(series)
    phi = np.zeros((1,2))
    # Value of 'r' in case of not normalized time series:
    r = r*np.std(series)

    for j in range(0,2):
        m = dim+j-1 # 'm' is the embbeding dimension used each iteration
        # Pre-definition of the varialbes for computational efficiency:
        patterns = np.zeros((m,N-m+1))
        aux = np.zeros((1,N-m+1))
        
        # First, we compose the patterns
        # The columns of the matrix 'patterns' will be the (N-m+1) patterns of 'm' length:
        if m == 1: # If the embedding dimension is 1, each sample is a pattern
            patterns = series
        else: # Otherwise, we build the patterns of length 'm':
            for i in range(m):
                patterns[i,:] = series[i:N-m+i]
        # We substract the baseline of each pattern to itself:
        for i in range(N-m+1):
            patterns[:,i] = patterns[:,i] - (np.mean(patterns[:,i]))
        
        # This loop goes over the columns of matrix 'patterns':
        # NOTE: May need to swap out these regular python math functions for the Numpy functions
        #       With input from NumPy arrays, the python math functions may be slower than Numpy's
        for i in range(N-m):
            if m == 1:
                dist = np.abs(patterns - np.tile(patterns[:,i],(1,N-m+1)))
            else:
                dist = np.max(np.abs(patterns - np.tile(patterns[:,i],(1,N-m+1))))
            # Second, we compute the maximum absolut distance between the
            # scalar components of the current pattern and the rest:
            # Third, we get the degree of similarity:
            simi = np.exp(((-1)*((dist)**n))/r)
            # We average all the degrees of similarity for the current pattern:
            aux[i] = (np.sum(simi)-1)/(N-m-1) # We substract 1 to the sum to avoid the self-comparison
        
        # Finally, we get the 'phy' parameter as the as the mean of the first
        # 'N-m' averaged drgees of similarity:
        phi[j] = np.sum(aux)/(N-m)

    # This is our FuzzyEn
    return np.log(phi[0]) - np.log(phi[1])

