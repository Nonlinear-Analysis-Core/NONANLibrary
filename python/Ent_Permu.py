import numpy as np

def Ent_Permu(data, m, tau):
    """
    (permEnt, hist) = Ent_Permu20180320(data, m, tau)
    inputs -  data: 1-D array of data being analyzed
              m: embedding dimension (order of permutation entropy) 
              tau: time delay
    outputs - permuEnt: value calculated using a log base of 2
              hist: number of occurences for each permutation order
    Remarks
    - It differs from the permutation entropy code found on MatLab Central in
      one way (see MathWorks reference). The code on MatLab Central uses the 
      log function (base e, natural log), whereas this code uses log2 (base 2
      ), as per Bandt & Pompe, 2002. However, this code does include a lag 
      (time delay) feature like the one on MatLab Central does.
    - Complexity parameters for time series based on comparison of 
      neighboring values. Based on the distributions of ordinal patterns, 
      which describe order relations between the values of a time series. 
      Based on the algorithm described by Bandt & Pompe, 2002.
    References
    - Bandt, C., Pompe, B. Permutation entropy: A natural complexity measure 
      for time series. Phys Rev Lett 2002, 88, 174102, 
      doi:10.1103/PhysRevLett.88.174102
    - MathWorks: http:www.mathworks.com/matlabcentral/fileexchange/
      37289-permutation-entropy)
    Jun 2016 - Created by Patrick Meng-Frecker, unonbcf@unomaha.edu
    Dec 2016 - Edited by Casey Wiens, email: unonbcf@unomaha.edu
    """
    def permutation_search(data, m, tau, N):
      permDict = dict()
      for cnt1 in range(N-tau*(m-1)):  # steps from 1 through length of data minus time delay multiplied by order minus 1
          permVal = np.argsort(data[cnt1:cnt1+tau*(m-1)+1:tau]).astype(str) # creates permutation of selected data range
          permVal = ''.join(permVal)  # concatenate array together as a string with no delimiter
          if permVal not in permDict:
              permDict[permVal] = 1
          else:
              permDict[permVal] += 1
      return np.array(list(permDict.values()))
    
    N = len(data)  # length of time series
    hist = permutation_search(data,m,tau,N)
    per = hist/np.sum(hist)	# ratio of each permutation vector match to total matches
    permEnt = np.negative(np.sum(np.multiply(per, np.log2(per))))   # performs entropy calucation
    return (permEnt, hist)



