import numpy as np
import numpy.matlib as matlib
import numpy.linalg as linalg
import sys # to implement args later
import copy
import warnings


def LyE_W(x, Fs, tau, dim, evolve):
  """
  inputs  - x, time series
          - Fs, sampling frequency
          - tau, time lag
          - dim, embedding dimension
          - evolve, parameter of the same name from Wolf's 1985 paper. This
            code expects a number of frames as an input.
  outputs - out, matrix detailing variables at each iteration
          - LyE, largest lyapunov exponent
  [out,LyE] = LyE_W20200820(X,Fs,tau,dim,evolve,SCALEMX,SCALEMN,ANGLMX,ZMULT)
          - SCALEMX, length of which the local structure of the attractor
            is no longer being probed
          - SCALEMN, length below which noise predominates the attractors
            behavior
          - ANGLMX, maximum angle used to constrain replacements
          - ZMULT, multiplier used to increase SCALEMX, unused in the
            current version of the code
  Remarks
  - This code calculates the largest lyapunov exponent of a time series
    according to the algorithm detailed in Wolf's 1985 paper. This code has
    been aligned with his code published on the Matlab file exchange in
    2016. It will largely find the same replacement points, the remaining
    difference being in the replacement algorithm.
  - The varargin can be used to specify some of the secondary parameters in
    the algorithm. All of the extra arguements must be specified if any are
    to be specified at all. Otherwise defaults are used.
  - ZMULT is not currently used in the code but was in a previous version.
    Its place in the subroutine inputs and outputs was kept in case it is
    put back in.
  - It should be noted that the process in the searching algorithm has a
    significant impact on the resulting LyE.
  - The code expects evolve to be the number of frames to use but we
    encourage you to report this as a time-value in publications.
  Prior - Created by Shane Wurdeman, unonbcf@unomaha.edu
        - Adapted by Brian Knarr, unonbcf@unomaha.edu
        - The code previously was influenced heavily by the FORTRAN syntax
          published in Wolf's 1985 paper. These were modified to better
          take advantage of MATLAB and speed up the code.
  Mar 2017 - Modified by Ben Senderling, unonbcf@unomaha.edu
           - Changed parameter "n" to "evolve."
           - Changed "ZMULT" back to 1.
           - Aligned the code with Wolf's Matlab File Exchange submission
             to find the same replacement points. This is now essential his
             algorithm but retains the speed of previous versions.
  Apr 2019 - Modified by Ben Senderling, unonbcf@unomaha.edu
           - Changed line 'range_exclude = range_exclude(range_exclude>=1 &
             range_exclude<=NPT);' to say '>=1' instead of '>1' to prevent
             self matches with the first point. This was indirectly
             accounted for by setting distances less than SCALEMN to 0.
           - '<SCALEMN' was removed from the code entirely and replaced with a
             '<=0'. This was checked against joint angles and EMG data. The
             change did not result in different pairs. This also removes an
            input.
  """
  
  x = np.array([x])
  SCALEMX = (np.max(x)-np.min(x))/10
  ANGLMX = 30*np.pi/180
  ZMULT = 1

  DT = 1/Fs

  ITS = 0
  distSUM = 0

  if np.size(x, axis=0) == 1:
    m = dim
    N = np.size(x, axis=1)
    M = N-(m-1)*tau
    Y = np.zeros((M,m))
    
    for i in range(0,m):
      Y[:,i]=x[:,(0+i*tau):(M + i*tau)]

    NPT=np.size(x, axis=1)-(dim-1)*tau-evolve # Size of useable data
    Y=Y[0:NPT+evolve,:] 

  else:
    Y=np.array(x)
    NPT=np.size(Y, axis=0)-evolve

  out=np.zeros((int(np.floor(NPT/evolve)+1),9),dtype="object")
  thbest=0
  OUTMX=SCALEMX

  # Find first pair
        
  # Distance from current point to all other points
  current_point = 0

  Yinit = matlib.repmat(Y[current_point],NPT,1)
  Ydiff = (Yinit - Y[0:NPT,:])**2
  Ydisti = np.sqrt(np.sum(Ydiff,1))
    
  # Exclude points too close on path and close in distance
  range_exclude = np.arange(current_point-10,current_point+10+1)
  range_exclude = range_exclude[(range_exclude>=0) & (range_exclude < NPT)]
  Ydisti[Ydisti<=0] = np.nan
  Ydisti[range_exclude] = np.nan
        
  # find minimum distance point for first pair
  current_point_pair = np.argsort(Ydisti)[0]

  for i in range(0, NPT, evolve):
    current_point = i
    # calculate starting and evolved distance
    if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
      start_dist = np.linalg.norm(Y[current_point,:] - Y[current_point_pair,:])
      end_dist = np.linalg.norm(Y[current_point+evolve,:] - Y[current_point_pair+evolve,:])
    else:
      start_dist = np.linalg.norm(Y[current_point,:] - Y[current_point_pair,:])
      end_dist = np.linalg.norm(Y[current_point+evolve,:] - Y[current_point_pair+evolve-1,:])


    
    
    # calculate total distance so far
    distSUM = distSUM + np.log2(end_dist/start_dist)/(evolve*DT) # DT is sampling rate?!
    ITS = ITS+1  # count iterations
    LyE=distSUM/ITS # max Lyapunov exponent
    
    #   CPP[i] = current_point_pair # Store found pairs
    
    out[int(np.floor(i/evolve))] = [ITS,current_point,current_point_pair,start_dist,end_dist,LyE,OUTMX,(thbest*180/np.pi),(ANGLMX*180/np.pi)]
    
    ZMULT=1
    
    if end_dist < SCALEMX:
      current_point_pair = current_point_pair+evolve
      if current_point_pair > NPT:
        current_point_pair = current_point_pair - evolve
        flag = 1
        (current_point_pair,ZMULT,ANGLMX,thbest,OUTMX) = get_next_point(flag,Y,current_point,current_point_pair,NPT,evolve,SCALEMX,ZMULT,ANGLMX)
      continue
    # find point pairing for next iteration
    flag=0
    (current_point_pair,ZMULT,ANGLMX,thbest,OUTMX) = get_next_point(flag,Y,current_point,current_point_pair,NPT,evolve,SCALEMX,ZMULT,ANGLMX)
    

  return (out, LyE)

def get_next_point(flag, Y, current_point, current_point_pair, NPT, evolve,SCALEMX, ZMULT, ANGLMX):

  # Distance from evolved point to all other points
  Yinit = np.matlib.repmat(Y[current_point+evolve,:],NPT,1)
  Ydiff = (Yinit - Y[0:NPT,:])**2
  Ydisti = np.sqrt(np.sum(Ydiff,axis=1))

  # Exclude points too close on path and close in distance than noise
  range_exclude = np.arange(current_point+evolve-10,current_point+evolve+10+1)
  range_exclude = range_exclude[(range_exclude >= 0) & (range_exclude < NPT)]
  Ydisti[range_exclude] = np.nan

  if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
    end_dist = np.linalg.norm(Y[current_point+evolve,:] - Y[current_point_pair+evolve,:])
  else:
    end_dist = np.linalg.norm(Y[current_point+evolve,:] - Y[current_point_pair+evolve-1,:])

  # Vector from evolved point to all other points
  Vnew = np.matlib.repmat(Y[current_point+evolve,:],NPT,1) - Y[:NPT,:]

  # Vector from evolved point to evolved point pair 
  if current_point_pair + evolve < len(Y) and current_point + evolve < len(Y):
    PT1 = Y[current_point+evolve,:]
    PT2 = Y[current_point_pair+evolve,:]
  else:
    PT1 = Y[current_point+evolve,:]
    PT2 = Y[current_point_pair+evolve-1,:]
  Vcurr = PT1-PT2

  # Angle between evolved pair vector and all other vectors
  # TODO: Had to add a summation here.
  cosTheta = np.abs(np.divide(np.sum(Vcurr.T*Vnew, axis=1),(Ydisti*end_dist)))
  theta = np.arccos(cosTheta)

  # Search for next point
  # -1 Meaning point not found.
  next_point=-1
  while next_point == -1:
    (next_point,ZMULT,ANGLMX,thbest,SCALEMX)=find_next_point(flag,theta,Ydisti,SCALEMX,ZMULT,ANGLMX)
  
  return next_point, ZMULT, ANGLMX, thbest, SCALEMX

def find_next_point(flag,theta,Ydisti, SCALEMX, ZMULT, ANGLMX):

  # Restrict search based on distance and angle
  PotenDisti= np.copy(Ydisti)
  PotenDisti[(Ydisti<=0) | (theta>=ANGLMX)] = np.nan

  next_point=-1
  if flag==0:
    next_point = np.argsort(PotenDisti)[0]
    # if closest angle point is within angle range -> point found and reset
    # search space
    if PotenDisti[next_point] <= SCALEMX:
      ANGLMX = 30*np.pi/180
      thbest=np.abs(theta[next_point])
      return (next_point, ZMULT, ANGLMX, thbest, SCALEMX)
    else:
      next_point=-1
      flag=1
  if flag == 1:
    PotenDisti=np.copy(Ydisti)
    PotenDisti[Ydisti<=0] = np.nan
    next_point = np.argsort(PotenDisti)[0]
    thbest=ANGLMX
  
  return (next_point, ZMULT, ANGLMX, thbest, SCALEMX)
