import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import sys

def LyE_R(X,Fs,tau,dim,*args):
    """
      inputs  - X, If this is a single dimentional array the code will use tau
                   and dim to perform a phase space reconstruction. If this is
                   a multidimentional array the phase space reconstruction will
                   not be used.
              - Fs, sampling frequency in units s^-1
              - tau, time lag
              - dim, embedding dimension
      outputs - out, contains the starting matched pairs and the average line
                     divergence from which the slope is calculated. The matched
                     paris are columns 1 and 2. The average line divergence is
                     column 3.
      [LyES,LyEL,out]=LyE_Rosenstein_FC(X,Fs,tau,dim,slope,MeanPeriod,plot)
     inputs  - slope, a four element array with the number of periods to find
                       the regression lines for the short and long LyE. This is
                       converted to indexes in the code.
              - MeanPeriod, used in the slope calculation to find the short and
                            long Lyapunov Exponents.
              - plot, a boolean specifying if a figure should be created
                      displaying the regression lines. This figure is visible
                      by default.
      outputs - LyES, short/local lyapunov exponent
              - LyEL, long/orbital lyapunov exponent
      Remarks
      - This code is based on the algorithm presented by Rosenstein et al,
        1992.
      - Recommendations for the slope input can be found in the references
        below. It is possible a long term exponent can not be found with your
        inputs. If your selection exceeds the length of the data LyEL will
        return as a NaN.
      Future Work
      - It may be possible to sped it up conciderably by re-organizing the for
        loops. A database for the matched points would need to be created.
      References
      - Rosentein, Collins and De Luca; "A practical method for calculating
        largest Lyapunov exponents from small data sets;" 1992
      - Yang and Pai; "Can stability really predict an impending slip-related
        fall among older adults?", 2014
      - Brujin, van Dieen, Meijer, Beek; "Statistical precision and sensitivity
        of measures of dynamic gait stability," 2009
      - Dingwell, Cusumano; "Nonlinear time series analysis of normal and
        pathological human walking," 2000
      Version History
      Jun 2008 - Created by Fabian Cignetti
               - It is suspected this code was originally written by Fabian
                 Cignetti
      Apr 2017 - Revised by Ben Senderling
               - Added comments section. Automated slope calculation. Added
                 calculation of orbital exponent.
      Jun 2020 - Revised by Ben Senderling
               - Incorporated the subroutines directly into the code since they
                 were only used in one location. Converted various for loops
                 into indexed operations. This significantly improved the
                 speed. Added if statements to compensate for errors with the
                 orbital LyE. If the data is such an orbital LyE would not be
                 found with the hardcoded regression line bounds. Made this 
                 slope and the file input optional. Removed the MeanPeriod as 
                 an imput and made it a calculation in the code. Added the out
                 array so the matched pairs and average line distance can be
                 reviewed, or used to finf the slope. Removed the progress
                 output to the command window since it was sped up
                 conciderably. Edited the figure output. Added code that allows
                 a multivariable input to be entered as X.
      Aug 2020 - Revised by Ben Senderling
               - Removed mean period calculation and turned it into an input.
                 This varies too widely between time series to have it
                 automatically calculated in the script. It was replaced with
                 tau to find paired points.
    """
    # Checked that X is vertically oriented. If X is a single or multiple
    # dimentional array the length is assumed to be longer than the width. It
    # is re-oriented if found to be different.
    X = np.array(X,ndmin=2)
    r,c = np.shape(X)
    if r > c:
        X = np.copy(X.transpose())

    # Checks if a multidimentional array was entered as X.
    if np.size(X,axis=0) > 1:
        M = np.shape(X)[1]
        Y=X
    else:
        # Calculate useful size of data
        N = np.shape(X)[1]
        M=N-(dim-1)*tau
        
        Y=np.zeros((M,dim))
        for j in range(dim):
            Y[:,j]=X[:,0+j*tau:M+j*tau]
    # Find nearest neighbors

    IND2=np.zeros((1,M),dtype=int)
    for i in range(M):
        # Find nearest neighbor.
        Yinit = np.matlib.repmat(Y[i],M,1)
        Ydiff = (Yinit-Y[0:M,:])**2
        Ydisti = np.sqrt(np.sum(Ydiff,axis=1))
        
        # Exclude points too close based on dominant frequency.
        range_exclude = np.arange(round((i+1)-tau*0.8-1),round((i+1)+tau*0.8))
        range_exclude = range_exclude[(range_exclude>=0) & (range_exclude<M)]
        Ydisti[range_exclude] = 1e5
        
        # find minimum distance point for first pair
        IND2[0,i] = np.argsort(Ydisti)[0]
        
    out = np.vstack((np.arange(M), np.ndarray.flatten(IND2)))

    # Calculate distances between matched pairs.
    DM = np.zeros((M,M))

    IND2len = np.shape(IND2)[1]

    for i in range(IND2len):
    # The data can only be propagated so far from the matched pair.
        EndITL=M-IND2[:,i][0]
        if (M-IND2[:,i][0])>(M-i):
            EndITL=M-i

        # Finds the distance between the matched paris and their propagated
        # points to the end of the useable data.
        DM[0:EndITL,i] = np.sqrt(np.sum((Y[i:EndITL+i,:]-Y[IND2[:,i][0]:EndITL+IND2[:,i][0],:])**2,axis=1))
        
    # Calculates the average line divergence.
    r,_ = np.shape(DM)

    AveLnDiv = np.zeros(len(DM))
    # NOTE: MATLAB version does not preallocate AveLnDiv, we could preallocate that.
    for i in range(r):
        distanceM = DM[i,:]
        if np.sum(distanceM) != 0:
            AveLnDiv[i]=np.mean(np.log(distanceM[distanceM>0])) 

    out = np.vstack((out,AveLnDiv))

    # Find LyES and LyEL
    plot = 0 # To avoid errors later on
    if len(sys.argv) == 0:
        output_list = out
    else:
        slope = args[0]
        MeanPeriod = args[1]
        plot = args[2]
        output_list = list()
        
        
        time = np.arange(0,len(AveLnDiv)) / Fs / MeanPeriod

        shortL = np.zeros(2,dtype=int)
        longL = np.zeros(2,dtype=int)
        
        # The values in slope are assumed to be the number of periods. These
        # are converted into indexes.
        if slope[0] == 0:
            shortL[0] = 0 # A value of 0 periods cannot be used.
        else:
            shortL[0] = round(slope[0]*MeanPeriod*Fs)
    
        shortL[1] = round(slope[1]*MeanPeriod*Fs)
        
        longL[0] = round(slope[2]*MeanPeriod*Fs)
        longL[1] = round(slope[3]*MeanPeriod*Fs)
        
        # If the index chosen exceeds the length of AveLnDiv then that exponent
        # is made a NaN.
        if shortL[1] <= np.size(np.nonzero(AveLnDiv)):
            slopeinterceptS=poly.polyfit(time[shortL[0]:shortL[1]+1], AveLnDiv[shortL[0]:shortL[1]+1],1)
            LyES=slopeinterceptS[1]
            timeS=time[shortL[0]:shortL[1]+1]
            LyESline=poly.polyval(timeS,slopeinterceptS)
        else:
            LyES=np.nan
        
        
        if longL[1] <= np.size(np.nonzero(AveLnDiv)):
            slopeinterceptL=poly.polyfit(time[longL[0]:longL[1]+1], AveLnDiv[longL[0]:longL[1]+1],1)
            LyEL=slopeinterceptL[1]
            timeL=time[longL[0]:longL[1]+1]
            LyELline=poly.polyval(timeL,slopeinterceptL)
        else:
            LyEL=np.nan
        
        output_list.append(LyES)
        output_list.append(LyEL)
        output_list.append(out)

    AveLnDiv = AveLnDiv[np.nonzero(AveLnDiv)]
    time = time[0:len(AveLnDiv)]
        
    # Plot data
        
    if plot == 1:
        plt.plot(time,AveLnDiv, color="black")
        plt.title("LyE")
        plt.xlabel("Periods (s)")
        plt.ylabel("<ln(divergence)>")
    
        if not np.isnan(LyES):
            plt.plot(timeS,LyESline,color="red",linewidth=3,label="LyE_Short = {}".format(LyES))
        if not np.isnan(LyEL):
            plt.plot(timeL,LyELline,color="green",linewidth=3,label="LyE_Long = {}".format(LyEL))

        plt.legend(loc="best")
        plt.show()
    return output_list