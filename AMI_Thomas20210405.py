import numpy as np
import scipy as sp

def AMI_Thomas20210405(x,L):
    """
    Usage: (tau,ami)=AMI_Thomas20210405(x,L)
    inputs:    x - time series, vertically orientedtrc files selected by user
               L - Maximum lag to calculate AMI until
    outputs:   tau - first true minimum of the AMI vs lag plot
               AMI - a vertically oriented vector containing values of AMI
               from a lag of 0 up the input L
    [ami]=AMI_Thomas20210405(x,y)
    inputs:   - x, single column array with the same length as y
              - y, single column array with the same length as x
    outputs   - ami, the average mutual information between the two arrays
    Remarks
    - This code uses a published method of calculating AMI to find an 
      acceptable lag with which to perform phase space reconstruction.
    - The algorithm is publically available at the citation below. Make sure
      to cite this work. The subroutine is fully their work.
    - In the case a value of tau could not be found before L the code will
      return an empty tau and the ami vector.
    - If it does find multiple values of tau but no definative minimum it
      will return all of these values.
    Future Work
    - None.
    References
    - Thomas, R. D., Moses, N. C., Semple, E. A., & Strang, A. J. (2014). An 
      efficient algorithm for the computation of average mutual information: 
      Validation and implementation in Matlab. Journal of Mathematical 
      Psychology, 61(September 2015), 45–59. 
      https://doi.org/10.1016/j.jmp.2014.09.001
    Sep 2015 - Adapted by Ben Senderling, email: bensenderling@gmail.com
                     Below I've set the code published by Thomas, Semple and
                     Strang to calculate AMI at various lags and to suggest
                     an appropriate tau.
    Apr 2021 - Modified by Ben Senderling, email bmchnonan@unomaha.edu
             - Modified in conjunction with NONAN validation efforts.
               Added the variable input arguements and second implementation.
    Validation
    
    Damped oscillator (approximate tau ~ 33)
    
    L=35
    t=(1:500)'
    a=0.005
    w=0.05
    x=exp(-a*t).*sin(w*t)
    
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
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR 
    PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # check size of input x
    (m,n) = np.shape(x)

    if m > 1 and n > 1:
        raise ValueError('Input vector is not one dimensional.')

    # calculate AMI at each lag
    ami = np.zeros((L,2))
    # fprintf('AMI: 00#')
    for i in range(L):
        ami[i,0] = i
        X = x[1:-1-i] # NOTE: Might be off by one.
        Y = x[i+1:]
        # NOTE: This takes it in as stacked rows, MATLAB assumes columns.
        ami[i,1] = average_mutual_information(np.vstack((X,Y)))

    tau = np.array([])
    #NOTE: Might be shape[1]
    for i in range(1,ami.shape[0]-1):
        if ami[i-1,1] >= ami[i,1] and ami[i,1] <= ami[i+1,1]:
            #NOTE: Axis might be wrong.
            tau = np.append(tau, ami[i,:])
    #NOTE: Testing new function that I've never used here, might be issues.
    ind = np.argmax(ami[:,1]<=(0.2*ami[0,1]))
    #NOTE: np.argmax returns a 0 if not found.
    if ind == 0:
        tau = np.append(tau, ami[ind,:])
    
    return (tau,ami)
    #TODO: Commenting this out, will work on later (4/28/2021), for the case that the input are two different column vectors
    # elseif nargin==2 && numel(varargin{1})>1 && numel(varargin{2})>1
        
    #     x = varargin{1}
    #     y = varargin{2}
    #     
    #     if numel(x)~=numel(y)
    #         error('x and y must be the same size')
    #     end
    #     ami=average_mutual_information([x,y])       
    #     varargout{1}=ami  
    # end

def average_mutual_information(data):
    """
    Usage: AMI = average_mutual_information(data) 
    Calculates average mutual information between 
    two 
    columns of data. It uses kernel density 
    estimation, 
    with a globally adjusted Gaussian kernel. 
    
    Input should be an n-by-2 matrix, with data sets 
    in adjacent 
    column vectors. 
    
    Output is a scalar.
    """
    # NOTE: Might be shape[0]
    n = data.shape[1]
    X = data[:,0]
    Y = data[:,1]
    # Example below is for normal reference rule in 
    # 2 dims, Scott (1992).
    hx = np.std(X)/(n**(1/6))
    hy = np.std(Y)/(n**(1/6))
    # Compute univariate marginal density functions. 
    P_x = univariate_kernel_density(X, X, hx) 
    P_y = univariate_kernel_density(Y, Y, hy) 
    # Compute joint probability density. 
    JointP_xy = bivariate_kernel_density(data, data, hx, hy) 
    AMI = np.sum(np.log2(np.divide(JointP_xy,np.multiply(P_x,P_y))))/n
    return AMI

def univariate_kernel_density(value, data, window):
    """
    Usage:  y = univariate_kernel_density(value, data, window) 
    Estimates univariate density using kernel 
    density estimation. 
    Inputs are: - value (m-vector), where density is estimated 
                - data (n-vector), the data used to estimate the density 
                - window (scalar), used for the width of density estimation. 
    Output is an m-vector of probabilities.
    """
    h = window 
    # NOTE: Make sure these are getting the correct lengths.
    n = len(data) 
    m = len(value) 
    # We use matrix operations to speed up computation 
    # of a double-sum. 
    prob = np.zeros((n,m))

    G = extended(value, n) 
    H = extended(data.T.copy(), m) 
    prob = sp.stats.norm.pdf((G-H)/h)
    fhat = np.sum(prob)/(n*h) 
    y = fhat
    return y

def bivariate_kernel_density(value, data, Hone, Htwo):
    """
    Usage: y = bivariate_kernel_density(value, data, Hone, Htwo) 
    Calculates bivariate kernel density estimates 
    of probability. 
    Inputs are: - value (m x 2 matrix), where density is estimated 
                - data (n x 2 matrix), the data used to estimate the density 
                - Hone (scalar) and Htwo (scalar) to use for the widths of density estimation. 
    Output is an m-vector of probabilities estimated at the values in ’value’. 
    """
    s = np.size(data)
    n = s[0]
    t = np.size(value) 
    number_pts = t[0] 
    rho_matrix = np.corrcoef(data)
    rho = rho_matrix[0,1]
    # The adjusted covariance matrix: 
    W = np.array([Hone**2,rho*Hone*Htwo,rho*Hone*Htwo,Htwo**2])
    differences = linear_depth(value,np.negative(data))

    prob = sp.stats.multivariate_normal(differences,mean=[0,0],cov=W) #  mu := [0,0], covariance := W
    cumprob = np.cumsum(prob)
    # NOTE: Next two lines may be incorrect.
    y = np.array([])
    y = np.append(y, (1/n)*cumprob[n])
    # NOTE: Not sure why i = i + 1 is included in all of these loops.
    for i in range(1,number_pts):
        index = n*i 
        y[i] = (1/(n))*(cumprob[index]-cumprob[index-n])
        i = i + 1 
    y = y.T.copy()
    return y

def linear_depth(feet, toes):
    """
    linear_depth takes a matrix ‘feet’ and lengthens 
    it in blocks, takes a matrix ‘toes’ and lengthens 
    it in Extended repeats, and then adds the
    lengthened ‘feet’ and ‘toes’ matrices to achieve 
    all sum combinations of their rows. 
    feet and toes have the same number of columns 
    """
    if feet.shape[1] == toes.shape[1]:
        a = feet.shape[0]
        b = toes.shape[0]

        blocks = np.zeros((a*b, toes.shape[1]))
        bricks = blocks 
        for i in range(a):
            blocks[i*b: i*b+1,:] = extended(feet[i,:],b) 
            bricks[i*b: i*b+1,:] = toes 
            i = i + 1 

    y = blocks + bricks 
    return y

def extended(vector, n):
    """
    Takes an m-dimensional row vector and outputs an 
    n-by-m matrix with n-many consecutive repeats of 
    the vector. Similarly,  it takes an 
    m-dimensional column vector and outputs an 
    m-by-n matrix. 
    Else, it returns the original input. 
    """
    M = vector 
    if vector.shape[0] == 1:
        M = np.zeros((n,len(vector)))
        for i in range(n):
            M[i,:] = vector

    if vector.shape[1] == 1:
        M = np.zeros((len(vector),n))
        for i in range(n):
            M[:,i] = vector

    y = M
    return y