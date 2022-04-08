import numpy as np

def FNN(data, tau, MaxDim, Rtol, Atol, speed):
  """
    data - column oriented time series
    tau - time delay
    MaxDim - maximum embedding dimension
    Rtol - threshold for the first criterion
    Atol - threshold for teh second criterion
    speed - a 0 for the code to calculate to the MaxDim or a 1 for the code
            to finish once a minimum is found
  Remarks
  - This code determines the embedding dimension for a time series using
    the false nearest neighbors method.
  - Recommended values are Rtol=15 and Atol=2;
  - Reference:   "Determining embedding dimension for phase-space
                  reconstruction using a geometrical construction",
                  M. B. Kennel, R. Brown, and H.D.I. Abarbanel,
                  Physical Review A, Vol 45, No 6, 15 March 1992,
                  pp 3403-3411.
  Future Work
  - Currently there are two methods of detecting a minimal percentage of
    false nearest neighbors. One method checks for a minima or zero
    percentage, the other looks for a limit. Currently only dim is
    returned. This code can be modified to use a comprimise of the two.
  Prior - Created by someone
  Feb 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
             No changes were made to the algorithm. Checks were added to
             provide information to the user in case of an error. The two
             methods described in future work were also modified to work
             cooperatively. In a previous version the second method (dim)
             overwrote the first method (dim2).
  Sep 2015 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
             Previously, dim was found after the for loop, this version has
             been modified to allow the code to find the minimum as it
             calculates FNN. This is set within the inputs.
             The check that was previously put in has been commented out.
  Oct 2015 - Modified by John McCamley, email: unonbcf@unomaha.edu
           - Embedded other required functions as subroutines.
  Mar 2017 - Modified by Ben Senderling, email: unonbcf@unomaha.edu
           - Removed global variables in favor of passing the variables
             from function to function directly. This significantly
             improved performance. Checked that the calculated percentages
             of nearest neighbors are the same as the previous version.
  May 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
           - Added if statement checkeding data orientation.
  Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
           - Changed indexing throughout so the input data array doesn't
             need to be reoriented. Changing this sped the code up an
             average 11% on 10 test signals.
           - Removed a couple small for loops and replaced with indexed
             operations. Was also able to remove within function and
             replaced with a single line of code.
           - Removed perviously commented out lines of code that were no
             longer used.
  """
  n = len(data) - tau * MaxDim
  data_array = np.array(data)
  RA = np.std(data_array)

  
  z = np.array([data_array[0:n]])
  y = np.array([[]])

  m_search = 2

  indx = np.array(np.arange(0,n))
  dim = np.array([])

  dE = np.zeros((MaxDim,1))

  for j in range(MaxDim):
    # y = np.array([y,z]) # adds additional dimension
    if j == 0:
      y = np.array([np.append(y,z)])
    else:
      y = np.array(np.vstack([y,z]))
    z = np.array([data_array[tau*(j+1):n+tau*(j+1)]])
    L = np.zeros((n))

    (y_model, z_model, sort_list, node_list) = kd_part(y,z,512)

    for i in range(len(indx)):
      yq = np.array(y[:, indx[i]]) # set up next point to look at

      b_upper = np.inf*np.ones(np.size(yq))
      b_lower = np.negative(b_upper)

      pqd = np.inf*np.ones((1,m_search))
      pqr = np.array([])
      pqz = np.array([])
      L_done = 0

      # A couple returning variables are not necessary. Check documentation of kd_search to see what they are.
      (pqd, y_model, z_model, _, _, pqz, _, _, sort_list, node_list) = kd_search(0,m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

      distance = pqz[0] - pqz[1]
      
      if np.abs(distance) > pqd[1]*Rtol:
        L[i] = 1
      
      if np.sqrt(pqd[1]**2 + distance**2)/RA > Atol:
        L[i] = 1

    dE[j] = np.sum(L)/n

    if speed == 1:
      if j >= 2 and ((dE[j-2]>dE[j-1] and (dE[j-1] < dE[j]))):
        dim = j-1
        break
      if j >= 1 and np.abs(dE[j] - dE[j-1]) <= 0.001:
        dim = j-1
        break
      if dE[j] == 0:
        dim = j
        break

  if speed == 0:
    for i in range(len(dE)):
      if np.abs(dE[i-1] - dE[i]) <= 0.001:
        dim = i-1
        break
      try: # In the case of where the code reaches to len(dE) - 1 and attempts to get dE[len(dE)], which causes an IndexError.
        if (dE[i] == 0) or ((dE[i-1] > dE[i] and (dE[i] < dE[i+1]))):
          dim = i
          break
      except: 
        break
  
  if np.size(dim) == 0:
    dim = MaxDim - 1
    print('No dimension found, dim set to MaxDim\n')

  # +1 because we started from zero and not one. Might change this in the future
  return (dE, dim+1)

def kd_part(y_in,z_in,bin_size):
  """
  Create a kd-tree and partitioned database for
  efficiently finding the nearest neighbors to a point
  in a d-dimensional space.
 
  y_in: original phase space data
  z_in: original phase space data corresponding to y_in
  bin_size: maximum number of distinct points for each bin
  The outputs are placed into global variables used by
  kdsearch and its subroutines.

  The outputs are...
  sort_list(:,1): discriminator: dimension to use in dividing data
  sort_list(:,2): partition: boundary for dividing data
  node_list(i,:): contains data for the i-th partition
  node_list(:,1): 1st element in y of this partition
  node_list(:,2): last element in y of this partition
  node_list(:,3): location in node_list of left branch
  node_list(:,4): location in node_list of right branch
  y_model: phase space data partitioned into a binary tree
  z_model: phase space data corresponding to each y_model point

  Algorithms from:
 
  "Data Structures for Range Searching", J.L. Bently, J.H. Friedman,
  ACM Computing Surveys, Vol 11, No 4, p 397-409, December 1979
 
  "An Algorithm for Finding Best Matches in Logarithmic Expected Time",
  J.H. Friedman, J.L. Bentley, R.A. Finkel, ACM Transactions on
  Mathematical Software, Vol 3, No 3, p 209-226, September 1977.

  Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
                - Formatted.
  """
  y_model = y_in
  z_model = z_in

  # d: dimension of phase space
  # n_y: number of points to put into partitioned database

  (d,n_y) = y_model.shape

  # Set up first node...
  node_list = np.array([[0, n_y, 0, 0]])
  sort_list = np.array([[0,0]], dtype=float)

  node = 0
  last = 0

  while node <= last: # check if the node can be divided
    segment = np.array([np.arange(node_list[node,0],node_list[node,1])])
    # previously: [node_list[node,1]:node_list[node,2]]
    rg = np.amax(y_model, axis=1) - np.amin(y_model, axis=1) # previously i and segment were swapped

    
    # segment.shape[1] is the length of the segment (specifically the length of the row)
    if np.max(rg) > 0 and segment.shape[1] >= bin_size: # it is divisible

      index = np.argsort(rg)
      yt = np.squeeze(y_model[:,segment], axis=1) # swapped : and segment
      zt = np.squeeze(z_model[:,segment], axis=1) 
      y_index = np.argsort(yt[index[d-1]])
      y_sort = np.sort(yt[index[d-1]]) # swapped : and index[d], 
      
      # estimate where the cut should go
      _,tlen = yt.shape 
          
      if np.fmod(tlen,2): # yt has an odd number of elements
        cut = y_sort[int((tlen+1)/2)]
      else: # yt has an even number of elements
        cut = (y_sort[int(tlen/2)]+y_sort[int(tlen/2+1)])/2
        # end of the median calculation
      
      L = y_sort <= cut
        
      if np.sum(L) == tlen: # then the right node will be empty...
        L = y_sort < cut  # ...so use a slightly different boundary
        cut = (cut+np.max(y_sort[L]))/2
        # end of the cut adjustment
        
        # adjust the order of the data
      y_model[:,segment] = np.expand_dims(yt[:,y_index], axis = 1)
      z_model[:,segment-1] = zt[:,y_index]
        
      # mark this as a non-terminal node
      sort_list[node,:] = [index[d-1], cut]
      node_list[node,2] = last + 1
      node_list[node,3] = last + 2
      last = last + 2
      
      # add the information for the new nodes
      node_list = np.vstack([node_list, [segment[0][0], segment[0][0] + np.sum(L)-1, 0, 0]]) 
      node_list = np.vstack([node_list, [segment[0][0] + np.sum(L), segment[0][tlen-1], 0, 0]]) 
      sort_list = np.vstack([sort_list, [[0,0],[0,0]]])
        
    # end of the splitting process
    
    node += 1   
    # end of the while loop

  return (y_model, z_model, sort_list, node_list) 


def kd_search(node, m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list):
  """
    [] = kdsearch(node)
    node - unknown
 
  Remarks
  - Search a kd_tree to find the nearest matches to the global variable
    yq, a vector.  The nearest matches will be put in the global variable
    pqr, and their distances in pqd.  See loclin_kd for a usage example.

  - pqd: Starts as an 1x2 with [inf, inf], then it is added to with the values from dist. 
    If pqd is longer than m_search, then only the values up to m_search are kept.
  - pqr: Starts as an empty array, then the data from y_model is added with the bounds
    defined by yi which is also defined by node_list. Then the last two data points are
    kept just like with pqd.
  - pqz: The same concept as pqr, except it is using data from the z_model.
  - index: Records the indices that are to be traversed to create the sorted order for pqd.
    This is used later on for pqr & pqz to order those to arrays by the order defined by index.
 
  Future Work
  - This code could be commmented to be understood easier.
 
  Feb 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
           - Commented and formated
  """
  if L_done == 1:
    return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

  if node_list[node,2] == 0: # it's a terminal node, so... 
    # first, compute the distances...
    yi = node_list[node,0:2] # index bounds of all y_model to consider
    yt = y_model[:,yi[0]:yi[1]]
    zt = z_model[:,yi[0]:yi[1]]

    d = len(yq) # get the dimension
    
    yq1 = yq # using this to swap the yq array back to a row vector
    yq = yq[:,np.newaxis] # column vector
    

    dist = np.sqrt(np.sum((yt[0:d,:]-yq[0:d])**2, axis = 0))

    yq = yq1

    # and then sort them and load pqd, pqr, and pqz

    # distances ^2
    pqd = np.append(dist, pqd)
    pqr = np.append(yt, pqr)
    pqz = np.append(zt,pqz)

    index = np.argsort(pqd) # distances sorted indexes
    pqd = np.sort(pqd) # distances sorted
    
    length = pqz.shape[0]
    
    if len(index) > length:  # to avoid indexing out of bounds
      pqr = pqr[index[0:len(pqz)]]
      pqz = pqz[index[0:len(pqz)]]
    else:                    # if index is <= length then we're okay to use the entirety of its length
      pqr = pqr[index]
      pqz = pqz[index]
    # keep only the first m_search points
    if len(pqd) > m_search:
      pqd = pqd[0:m_search]   
 
    length = pqz.shape[0]
    
    # most of the time this is true
    if length > m_search:
        pqr = pqr[0:m_search]
        pqz = pqz[0:m_search]
    
    # m_search-1 to not cause indexing issues
    if any((np.abs(yq-b_lower)<=pqd[m_search-1]) | (np.abs(yq-b_upper)<=pqd[m_search-1])): 
        L_done = 1
    
    return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)
  else: # it's not a terminal node, so search a little deeper
    disc = int(sort_list[node,0])
    part = sort_list[node,1]

    if yq[disc] <= part: # determine which child node to go to
        
      temp = b_upper[disc]
      b_upper[disc] = part
      #         [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd)
      (pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list) = kd_search(node_list[node,2],m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list)
      b_upper[disc] = temp
    else:
      temp = b_lower[disc]
      b_lower[disc] = part
      #         [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd)
      (pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list) = kd_search(node_list[node,3],m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list)   
      b_lower[disc] = temp
            
    if L_done:
        return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)
    
    if yq[disc] <= part: # determine whether other child node needs to be searched
      temp = b_lower[disc]
      b_lower[disc] = part
        
      L = overlap(yq,m_search,pqd,b_upper,b_lower)
      if L == 1:
        #             [pqd]=kdsearch(node_list(node,4),m_search,yq,pqd);
        (pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list) = kd_search(node_list[node,4],m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
                    
        b_lower[disc] = temp   
    else:
      temp = b_upper[disc]
      b_upper[disc] = part
        
      L = overlap(yq,m_search,pqd,b_upper,b_lower)
      if L == 1:
        #             [pqd]=kdsearch(node_list(node,3),m_search,yq,pqd);
        (pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list) = kd_search(node_list[node,3],m_search,yq,pqd,y_model,z_model,L_done,pqr,pqz,b_upper,b_lower,sort_list,node_list);
                    
        b_upper[disc] = temp
            
    if L_done:
        return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)
  
  return (pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

def overlap(yq, m_search, pqd, b_upper, b_lower):
  """
  L = overlap
  Inputs: yq, m_search, pqd, b_upper, b_lower.
  Outputs: L - unknown

  Remarks
  - References: - "Data Structures for Range Searching", J.L. Bently, J.H.
                  Friedman, ACM Computing Surveys, Vol 11, No 4, p 397-409,
                  December 1979.
                - "An Algorithm for Finding Best Matches in Logarithmic
                  Expected Time", J.H. Friedman, J.L. Bentley, R.A. Finkel,
                  ACM Transactions on Mathematical Software, Vol 3, No 3,
                  p 209-226, September 1977.
 
  Future Work
  - None.
 
  Mar 2015 - Modified by Ben Senderling, phone 267-980-0425, email bensenderling@gmail.com
  """
  # indexing with m_search = 2 on a 2 size numpy array causes errors so it is pqd[m_search-1]
  dist = pqd[m_search-1]**2
  sum = 0

  for i in range(len(yq)):
    
    if yq[i] < b_lower[i]:
        sum = sum + (yq[i]-b_lower[i])**2
        if sum > dist:
            L = 0
            return L
        # end of the sum > dist
    elif yq[i] > b_upper[i]:
        sum = sum + (yq[i]-b_upper[i])**2
        if sum > dist:
            L = 0
            return L
        # end of the sum > dist if
    # end of the yq(i) <> a bound if  
  # end of the i loop

  L = 1

  return L