import numpy as np

def Ent_Symbolic(X, L):
    """
    SymEnt = Ent_Symbolic20180320(X, L)
    symbolicEnt Calculates the Symbolic Entropy for given data.
    Input -   X: 1-Dimensional binary array of data
              L: Word length
    Output -  NCSE: Normalized Corrected Shannon Entropy
    Remarks
    - This code calculates the Symbbolic Entropy value for the provided data
      at a given word length described by - Aziz, W., Arif, M., 2006.
      "Complexity analysis of stride interval time series by threshold
      dependent symbolic entropy." Eur. J. Appl. Physiol. 98: 30-40.
    Jun 2017 - Created by William Denton, unonbcf@unomaha.edu
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    words = np.zeros(X.shape[0] - L+1)

    str_rep = str(np.apply_along_axis(lambda row: row.astype('|S1').tobytes().decode('utf-8'),
                axis=0,
                arr=X))
    for i in range(X.shape[0]-L+1):
        words[i] = int(str_rep[i:i+L],2)
    
    max_words = 2**L
    P = np.zeros(max_words)
    H = np.zeros(max_words)

    for i in range(max_words):
        P[i] = np.where(words == i)[0].size/words.size
        Hval = P[i]*np.log2(P[i])
        if np.isnan(Hval):
            pass
        else:
            H[i] = Hval

    H = np.negative(np.sum(H))
    
    So = np.unique(words).size
    Sm = max_words
    CSE = H+(So-1) / (2*Sm*np.log(2))
    CSEm = -np.log2(1/Sm) + (Sm-1) / (2*Sm*np.log(2))
    return CSE/CSEm