import numpy as np

def Ent_Ap(data, dim, r):
    """
    Ent_Ap20120321
      data : time-series data
      dim : embedded dimension
      r : tolerance (typically 0.2)
    
      Changes in version 1
          Ver 0 had a minor error in the final step of calculating ApEn
          because it took logarithm after summation of phi's.
          In Ver 1, I restored the definition according to original paper's
          definition, to be consistent with most of the work in the
          literature. Note that this definition won't work for Sample
          Entropy which doesn't count self-matching case, because the count 
          can be zero and logarithm can fail.
    
      *NOTE: This code is faster and gives the same result as ApEn = 
             ApEnt(data,m,R) created by John McCamley in June of 2015.
             -Will Denton
    
    ---------------------------------------------------------------------
    coded by Kijoon Lee,  kjlee@ntu.edu.sg
    Ver 0 : Aug 4th, 2011
    Ver 1 : Mar 21st, 2012
    ---------------------------------------------------------------------
    """


    r = r*np.std(data)
    N = len(data)
    phim = np.zeros(2)
    for j in range(2):
        m = dim+j
        phi = np.zeros(N-m+1)
        data_mat = np.zeros((N-m+1,m))
        for i in range(m):
            data_mat[:,i] = data[i:N-m+i+1]
        for i in range(N-m+1):
            temp_mat = np.abs(data_mat - data_mat[i,:])
            AorB = np.unique(np.where(temp_mat > r)[0])
            AorB = len(temp_mat) - len(AorB)
            phi[i] = AorB/(N-m+1)
        phim[j] = np.sum(np.log(phi))/(N-m+1)
    AE = phim[0] - phim[1]
    return AE


