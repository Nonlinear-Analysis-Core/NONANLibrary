import numpy as np

def Surr_PseudoPeriodic(y,tau,dim,rho):
    """
    inputs  - y, time series
            - tau, time lag for phase space reconstruction
            - dim, embedding dimension for phase space reconstruction
            - rho, noise radius
    outputs - ys, surrogate time series
            - yi, selected indexes for surrogate from original time series
    Remarks
    - This code produces one pseudo periodic surrogate. It is appropriate to
      run on period time series to remove the long-term correlations. This is
      useful when testing for the presense of chaos or testing various
      nonlinear analysis methods.
    - There may be an optimal value of rho. This can be found by using a
      different function. Or it can be specified manually.
    - If rho is too low, ~<0.01, the code will not be able to find a
      neighbor.
    Future Work
    - Previous versions had occationally created surrgates with plataues. It
      is unknown if these are present in the current version.
    References
    - Small, M., Yu, D., & G., H. R. (2001). Surrogate Test for 
      Pseudoperiodic Time Series Data. Physical Revew Letters, 87(18). 
      https://doi.org/10.1063/1.1487534
    Version History
    May 2001 - Created by Michael Small
             - The original version of this script was converted from
               Michael Small's C code to MATLAB by Ben Senderling.
    Jun 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
             - The original was heavily modified while referencing Small, 
               2001. For loops and equations were indexed to save space and 
               speed up the script. The phase space reconstruction was 
               changed from a backwards to forwards lag. The initial seed was
               removed as an input. Added a line to remove self-matches.
               Added an exception in case a new value of xi could not be
               found.
    Mar 2021 - Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Tried to fix the problem of new points not being able to be
               found.
    """
    # Phase space reconstruction

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    N = len(y)
    Y = np.zeros((N-(dim-1)*tau,dim))
    for i in range(dim):
        Y[:,i] = y[i*tau:N-(dim-(i+1))*tau]

    # Seeding and initial points
    lenY = Y.shape[0]
    xi = int((np.floor(np.random.rand(1)*lenY)+1)[0])
    ys = np.zeros((lenY,1))
    ys[0] = y[xi]
    yi = np.zeros((lenY,1))
    yi[0] = xi

    M = lenY-2

    # Construct the surrogate
    for i in range(1,lenY): # steps of one as well.
        
        # Calculates the distance from the previous point to all other points.
        # This is the probability calculation in Small, 2001. Points that are 
        # close neighbors will end up with a higher value.
        prob = np.exp(np.negative(np.sqrt(np.sum(np.power(Y[:M,:]-np.matlib.repmat(Y[xi,:],M,1),2),axis=1)))/rho)
        # A self-match will be exp(0)=1, which can be large compared to the
        # other values. It could be removed. Adding in this line appears to
        # produce decent surrogates but makes the optimization method
        # un-applicable.
        #     prob(xi)=0
        # Cummulative sum of the probability
        sum3=np.cumsum(prob)
        # A random number is chosen between 0 and the cummulative probability.
        # Where it goes above the cumsum that is chosen as the next point, +2.
        # Most of the values in prob have a very small value, the close
        # neighbors are the spikes.
        xi_n= np.array([])
        ind=0
        while xi_n.size == 0:
            a = (np.random.rand(1))[0]
            xi_n = np.where(sum3<(a*sum3[-1]))[0]
            if xi_n.size > 0:
                xi_n = xi_n[-1]+2
            ind=ind+1
            if xi_n == xi and ind == 100:
                xi_n = xi_n+1
                break
            elif ind>100:
                raise Exception('a new value of xi could not be found, check that rho is not too low')
        xi=xi_n
        
        # Add the new point to the surrogate time series.
        ys[i] = y[xi]
        yi[i] = xi
        
    return (ys, yi)



