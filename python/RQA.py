import warnings, sys, string
import numpy as np
from skimage.transform import rotate
from sklearn.metrics import pairwise_distances
from scipy import stats as st
from scipy.spatial import distance as spd
import matplotlib.pyplot as plt

def RQA(DATA,TYPE,EMB,DEL,ZSCORE,NORM,LINELENGTH,SETPARA,SETVALUE,PLOTOPTION,nargout):
    """
    Usage: (RP, RESULTS)=RQA20210210(DATA,TYPE,EMB,DEL,ZSCORE,NORM,LINELENGTH,SETPARA,SETVALUE,PLOTOPTION)
    Inputs  - DATA, a double-variable with each dimension of the
                    to-be-analyzed signal as a row of numbers in a separate
                    column. If too many columns are present for the TYPE of
                    analysis selected, the other columns will be ignored
                    (i.e. for 'cRQA' only the first two columns will be
                    used).
            - TYPE, a string indicating which type of RQA to run (i.e.
                    'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
                    TYPE = 'RQA'.
            - EMB, the number of embedding dimensions (i.e., EMB = 1 would
                   be no embedding via time-delayed surrogates, just using
                   the provided number of colums as dimensions. The default
                   value is EMB = 1.
            - DEL, the delay parameter used for time-delayed embedding (if
                   EMB > 1). The default value is DEL = 1.
            - ZSCORE, indicates, whether the data (i.e., the different
                   columns of DATA, being the different signals or
                   dimensions of a signal) should be z-scored before
                   performing MdRQA:
                   0 - no z-scoring of DATA
                   1 - z-score columns of DATA
                   The default value is ZSCORE = 0.
            - NORM, the type of norm by with the phase-space is normalized.
                   The following norms are available:
                   'euc' - Euclidean distance norm
                   'max' - Maximum distance norm
                   'min' - Minimum distance norm
                   'non' - no normalization of phase-space
                   The default value is NORM = 'non'.
            - SETPARA, the parameter which you would like to set a target
                   value for the recurrence plot (i.e. 'radius' or
                   'recurrence'). The default value is SETPARA = 'radius'.
            - SETVALUE, sets the value of the selected parameter. If
                   SETVALUE = 1, then the radius will be set to 1 if SETPARA
                   = 'radius' or the radius will be adjusted until the
                   recurrence is equal to 1 if SETPARA = 'recurrence'. The
                   default value if SETPARA = 'radius' is 1. The default
                   value if SETPARA = 'recurrence' is 2.5.
            - PLOTOPTION, a 1 will display a plot, 0 will not display it.
            - nargout, a number of arguments we want to return from this function.
    Outputs - RP is a matrix holding the resulting recurrence plot.
            - RESULTS is a dictionary holding the following recurrence
              variables:
              1.  DIM    - dimension of the input data (used for mdRQA)
              2.  EMB    - embedding dimension used in the calculation of the
                           distance matrix
              3.  DEL    - time lag used in the calculation of the distance
                           matrix
              4.  RADIUS - radius used for the recurrence plot
              5.  NORM   - type of normilization used for the distance matrix
              6.  ZSCORE - whether or not zscore was used
              7.  Size   - size of the recurrence plot
              8.  %REC   - percentage of recurrent points
              9.  %DET   - percentage of diagonally adjacent recurrent points
              10. MeanL  - average length of adjacent recurrent points
              11. MaxL   - maximum length of diagonally adjacent recurrent
                           points
              12. EntrL  - Shannon entropy of distribution of diagonal lines
              13. %LAM   - percentage of vertically adjacent recurrent points
              14. MeanV  - average length of diagonally adjacent recurrent
                           points
              15. MaxV   - maximum length of vertically adjacent recurrent
                           points
              16. EntrV  - Shannon entropy of distribution of vertical lines
              17. EntrW  - Weighted entropy of distribution of vertical
                           weighted sums
    Remarks
    - Computes a recurrence plot for either recurrence quantification
      analysis (RQA), cross recurrence quantification analysis (cRQA), joint
      recurrence quantification analysis (jRQA), or multidimensional
      recurrence quantification analysis (mdRQA). Either radius or target
      recurrence can be set.
    Reference:
    - Wallot, S., Roepstorff, A., & Monster, D. (2016). Multidimensional
      Recurrence Quantification Analysis (MdRQA) for the analysis of
      multidimensional time-series: A software implementation in MATLAB and
      its application to group-level data in joint action. Frontiers in
      Psychology, 7, 1835. http://dx.doi.org/10.3389/fpsyg.2016.01835
    - Eroglu, D., Peron, T. K. D., Marwan, N., Rodrigues, F. A., Costa, L. D.
      F., Sebek, M., ... & Kurths, J. (2014). Entropy of weighted recurrence
      plots. Physical Review E, 90(4), 042919.
    
    Jul 2016 Modified by Sebastian Wallot
             - VERSION 1.0.0
               28. July 2016 by Sebastian Wallot, Max Planck Insitute for
               Empirical Aesthetics, Frankfurt, Germany & Dan M?nster, Aarhus
               University, Aarhus, Denmark
    Jul 2017 Modified by Will Denton
             - VERSION 1.1.0
               06. July 2017 by Will Denton (wdenton@unomaha.edu), Troy Rand
               (troyrand@gmail.com), and Casey Wiens (cwiens32@gmail.com),
               Biomechanics Research Building, University of Nebraska at
               Omaha. Changes include cleaning up some errors, making the
               default input arguments function correctly, incorperating
               other types of RQA (e.g. RQA, CRQA, JRQA), allowing %REC to be
               set instead of radius, incorporating weighted recurrence
               plots, and adding weighted entropy.
    May 2019 Modified by Will Denton
             - VERSION 1.1.1 (05/09/2019)
             - Added updates/patching.
             - Added usage tracking to allow the Department of Biomechanics
               at the University of Omaha see which codes and versions are
               being used.
             - Added error reporting to allow the Department of Biomechanics
               at the University of Omaha to make improvements to this code.
    Nov 2019 Modified by Will Denton
             - VERSION 1.1.2 (11/18/2019)
             - Fixed usage and error reporting to work with MacOS.
             - Fixed left plot to align with the recurrence plot when zoomed
               in and panning around.
    Jul 2020 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Removed automatic update code and version history code.
    Dec 2020 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Commented out waitbar and added PLOTOPTION input to control
               figure creation.
    Feb 2021 Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Made line length an input with a default of 1.
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

    # Set default parameters if no input exists
    # If SETPARA is not specified, set to 'radius'

    DATA = np.array(DATA)
    
    if SETPARA == None:
        SETPARA = 'radius'


    # If SETVALUE is not specified, set to 1 if radius is set or 2.5 if perRec is set
    
    if SETVALUE == None:
        if SETPARA == 'radius' or SETPARA == 'rad' or SETPARA == 1:
            radius=1
            runSetRad=0
        elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
            radiusStart = 0.01
            radiusEnd = 0.5
            runSetRad = 1
            SETVALUE = 2.5    
    else:
        if SETPARA == 'radius' or SETPARA =='rad' or SETPARA == 1:
            radius = SETVALUE
            runSetRad = 0
        elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
            radiusStart = 0.01
            radiusEnd = 0.5
            runSetRad = 1

    # If LINELENGTH is not specified, set to '1'
    if LINELENGTH == None:
        LINELENGTH = 1
    # If NORM is not specified, set to 'non'
    if NORM == None:
        NORM = 'non'
    # If ZSCORE is not specified, set to 0
    if ZSCORE == None:
        ZSCORE = 0
    # If DEL is not specified, set to 1
    if DEL == None:
        DEL = 1
    # If EMB is not specified, set to 1
    if EMB == None:
        EMB = 1
    # If EMB is not specified, set to 1
    if TYPE == None:
        TYPE = 'RQA'
    # If z score is selected then z score the data
    if ZSCORE: # == 1
        DATA = st.zscore(DATA)

    # Set DIM and select proper column(s) of data if too many exist
    # if 1 dimensional, only one column, else column is reflected by the second value of the tuple from shape.
    if DATA.ndim == 1:
        r = 1
    else:
        r = np.shape(DATA)[0]

    TYPE = TYPE.upper()
    
    if TYPE == 'RQA':
        TYPE = 'RQA'
    elif TYPE == 'CRQA' or TYPE == 'CROSS':
        TYPE = 'CRQA'
    elif TYPE == 'JRQA' or TYPE == 'JOINT':
        TYPE = 'JRQA'
    elif TYPE == 'MDRQA' or TYPE == 'MD' or TYPE == 'MULTI':
        TYPE = 'MDRQA'
    

    if TYPE == 'RQA':
        DIM = 1
        if r > 1:
            DATA = DATA[:,0]
            warnings.warn("More than one column of data. Only using first column.")
    elif TYPE == 'CRQA':
        DIM = 2
        if r > 2:
            DATA = DATA[:,:2]
            warnings.warn("More than two columns of data Only using first two columns.")
    elif TYPE == 'JRQA':
        DIM = r
        if r < 2:
            raise Exception("Input data must have at least two columns.")
    elif TYPE == 'MDRQA':
        DIM = r

    # Embed the data
    if EMB > 1:
        # Preallocation is to be implemented.
        # if r > 1:
        #     tempDATA = np.zeros((DIM*EMB,DATA.shape[1]-(EMB-1)*DEL))
        # else:
        #     tempDATA = np.zeros((DIM*EMB,DATA.shape[0]-(EMB-1)*DEL))
        onerow = False
        for i in range(EMB):
            if i == 0:
                try:
                    tempDATA = np.array(DATA[:,i*DEL:DATA.shape[1]-(EMB-i)*DEL+1])
                except:
                    onerow = True
                    tempDATA = np.expand_dims(np.array(DATA[i*DEL:DATA.shape[0]-(EMB-i)*DEL+1]),axis=0)
            elif onerow:
                tempDATA = np.concatenate((tempDATA, np.array([DATA[i*DEL:DATA.shape[0]-(EMB-i)*DEL+1]])),axis=0)
            else:
                tempDATA = np.concatenate((tempDATA, DATA[:,i*DEL:DATA.shape[1]-(EMB-i)*DEL+1]),axis=0)

        DATA = tempDATA
        tempDATA = np.delete(tempDATA,np.s_[::]) # deletes everything from array.

    a = [i for i in range(r)]    
    if TYPE == 'RQA':
        if EMB > 1:
            pairs = DATA.T.copy()
            a[0] = pairwise_distances(pairs,metric='euclidean')
            a[0] = np.abs(a[0])*-1 # make values negative
        else:
            a[0] = pairwise_distances(DATA.reshape(-1,1),metric='euclidean')
            a[0] = np.abs(a[0])*-1 # make values negative
    elif TYPE == 'CRQA':
        # Euclidean distance between the first and second column
        if EMB > 1:
            # indexes our pairs from the rows of the data by steps of DIM (for when EMB > 1)
            pairs1 = DATA[np.s_[::DIM],:].T.copy()
            pairs2 = DATA[np.s_[1::DIM],:].T.copy()

            a[0] = pairwise_distances(pairs1,pairs2,metric='euclidean')
            a[0] = np.abs(a[0])*-1
        else:
            a[0] = pairwise_distances(DATA[0].reshape(-1,1), DATA[1].reshape(-1,1), metric='euclidean')
            a[0] = np.abs(a[0])*-1
        # doing this to avoid unneccesarily creating a weighted recurrence plot later on.
        a = [a[0]]
    elif TYPE == 'JRQA':
        for i in range(r):
            if EMB > 1:
                pairs = DATA[np.s_[i::DIM],:].T.copy()
                a[i] = pairwise_distances(pairs, metric='euclidean')
                a[i] = np.abs(a[i])*-1
            else:
                a[i] = pairwise_distances(DATA[i,:].reshape(-1,1), DATA[i,:].reshape(-1,1), metric='euclidean')
                a[i] = np.abs(a[i])*-1
    elif TYPE == 'MDRQA':
        a[0] = pairwise_distances(DATA.T,metric='euclidean')
        a[0] = np.abs(a[0])*-1
        a = [a[0]]

    # Normalize distance matrix
    if 'euc' in NORM:
        for i in range(len(a)):
            b = np.mean(a[i][np.where(a[i]<0)])
            b = np.negative(np.sqrt(np.abs((b**2)+2*(DIM*EMB))))
            a[i] = a[i]/np.abs(b)
    elif 'min' in NORM:
        for i in range(len(a)):
            b = np.max(a[i][np.where(a[i]<0)])
            a[i] = a[i]/np.abs(b)
    elif 'max' in NORM:
        for i in range(len(a)):
            b = np.min(a[i][np.where(a[i]<0)])
            a[i] = a[i]/abs(b)
    elif 'non' in NORM:
        pass # do nothing
    else:
        raise Exception('No appropriate norm parameter specified.')

    # Create weighted recurrence plot
    index = -1
    for i in range(len(a)-1):
        a[i+1] = np.multiply(a[i],a[i+1])
        index = i
    if index == 0:
        a = np.negative((np.abs(a[index+1]))**(1/(index+2)))

    if TYPE in 'RQA' or TYPE in 'CRQA' or TYPE in 'MDRQA':
        a = a[0]

    # Calculate recurrence plot
    if SETPARA == 'radius' or SETPARA == 'rad' or SETPARA == 1:
        perRec = rqaPerRec(a, TYPE, radius, LINELENGTH)
        (diag_hist, vertical_hist) = rqaHistograms(a.copy(), DATA, TYPE, radius)
    elif SETPARA == 'perrec' or SETPARA == 'recurrence' or SETPARA == 2:
        (perRec, radius) = setRadius(a, TYPE, radiusStart, SETVALUE, radiusEnd, LINELENGTH)
        (diag_hist, vertical_hist) = rqaHistograms(a.copy(), DATA, TYPE, radius)

    RESULTS = {
        "DIM" : DIM,
        "EMB" : EMB,
        "DEL" : DEL,
        "RADIUS" : radius,
        "NORM" : NORM,
        "ZSCORE" : ZSCORE,
        "SIZE" : len(a),
        "REC" : perRec
    }
    if RESULTS["REC"] > 0:
        RESULTS["DET"] = 100 * np.sum(diag_hist[diag_hist>LINELENGTH])/np.sum(diag_hist)
        RESULTS["MeanL"] = np.mean(diag_hist[diag_hist>LINELENGTH])
        RESULTS["MaxL"] = np.max(diag_hist)
        # Create our histogram using data points that are greater than LINELENGTH (Default is 1 for LINELENGTH)
        # Our bins to use is a range from the minimum to the maximum
        (count, bins) =  np.histogram(diag_hist[diag_hist>LINELENGTH])
        total = np.sum(count)
        p = np.divide(count, total)
        zero_indices = np.where(count == 0)
        p = np.delete(p, zero_indices)
        RESULTS["EntrL"] = np.negative(np.sum(np.multiply(p,np.log2(p))))
        RESULTS["LAM"] = 100*np.sum(vertical_hist[vertical_hist>LINELENGTH])/np.sum(vertical_hist)       
        RESULTS["MeanV"] = np.mean(vertical_hist[vertical_hist>LINELENGTH])
        RESULTS["MaxV"] = np.max(vertical_hist)
        (count, bins) =  np.histogram(vertical_hist[vertical_hist>LINELENGTH])
        total = np.sum(count)
        p = np.divide(count, total)
        zero_indices = np.where(count == 0)
        p = np.delete(p, zero_indices)
        RESULTS["EntrV"] = np.negative(np.sum(np.multiply(p,np.log2(p))))
        RESULTS["EntrW"] = RQA_WeightedEntropy(a)
    else:
        RESULTS["DET"] = np.nan
        RESULTS["MeanL"] = np.nan
        RESULTS["MaxL"] = np.nan
        RESULTS["EntrL"] = np.nan
        RESULTS["LAM"] = np.nan
        RESULTS["MeanV"] = np.nan
        RESULTS["MaxV"] = np.nan
        RESULTS["EntrV"] = np.nan
        RESULTS["EntrW"]  = np.nan

    a[a >= np.negative(radius)] = 1.
    a[a < np.negative(radius)] = 0.

    RP = rotate(1. - a, 90)

    if nargout == 1:
        out = [RESULTS]
    elif nargout == 2:
        out = [RESULTS,RP]

    if PLOTOPTION > 0:
       
        title = "DIM = {}, EMB = {}, DEL = {}, RAD = {:.5f}, NORM = {}, ZSCORE = {}".format(DIM,EMB,DEL,radius,NORM,ZSCORE)
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(5, 5, hspace=.7, wspace=.7)
        main_plot = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], xticklabels=[],aspect='auto')
        x_plot = fig.add_subplot(grid[-1, 1:], yticklabels=[],aspect='auto')
        main_plot.imshow(RP, cmap='gray')
        main_plot.set_title(title,fontsize='x-small')
        main_plot.set_xticks([])
        main_plot.set_yticks([])
        main_plot.set_xlabel('X(i)',fontsize='large')
        main_plot.set_ylabel('Y(j)',fontsize='large')
        x_plot.set_xmargin(0)
        x_plot.set_ymargin(0)
        y_plot.set_xmargin(0)
        y_plot.set_ymargin(0)
        if TYPE in 'RQA' or TYPE in 'MDRQA':
            if DATA.ndim == 1: DATA = np.expand_dims(DATA,axis=0)
            x_plot.plot(np.arange(DATA.shape[1]),DATA[0],'-k')
            y_plot.plot(np.flip(DATA[0]),np.arange(DATA.shape[1],0,-1),'-k')
        elif TYPE in 'CRQA':
            x_plot.plot(np.arange(DATA.shape[1]),DATA[0,:], 'k-')
            y_plot.plot(np.flip(DATA[1,:]),np.arange(DATA.shape[1],0,-1),'-k') #HACK: Might be plotting backward.
        elif TYPE in 'JRQA':
            # TODO: MATLAB side shows only one line along the x-axis, our plot ends up plotting more than one line... which one is correct?
            y_plot.plot(np.flip(DATA[0,:]),np.arange(DATA.shape[1]),'-k')
            y_plot.invert_yaxis()
            for i in range(r):
                x_plot.plot(np.arange(DATA.shape[1]),DATA[i,:], 'k-')     
        fig.text(.02,.24,"%REC = {:.2f}".format(RESULTS["REC"]))
        fig.text(.02,.21,"%DET = {:.2f}".format(RESULTS["DET"]))
        fig.text(.02,.18,"MaxL = {:.0f}".format(RESULTS["MaxL"]))
        fig.text(.02,.15,"MeanL = {:.2f}".format(RESULTS["MeanL"]))
        fig.text(.02,.12,"EntrL = {:.2f}".format(RESULTS["EntrL"]))
        fig.text(.02,.09,"%LAM = {:.2f}".format(RESULTS["LAM"]))
        fig.text(.02,.06,"MaxV = {:.0f}".format(RESULTS["MaxV"]))
        fig.text(.02,.03,"MeanV = {:.2f}".format(RESULTS["MeanV"]))
        fig.text(.02,.0,"EntrV = {:.2f}".format(RESULTS["EntrV"]))
        if PLOTOPTION == 1:
            plt.show()
    # TODO: Tabs are something that matplotlib does not do by default, if needed this tab feature may be implemented later.
    if PLOTOPTION > 1:
        title = "DIM = {}, EMB = {}, DEL = {}, RAD = {:.5f}, NORM = {}, ZSCORE = {}".format(DIM,EMB,DEL,radius,NORM,ZSCORE)
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(5, 5, hspace=.7, wspace=.7)
        main_plot = fig.add_subplot(grid[:-1, 1:])
        y_plot = fig.add_subplot(grid[:-1, 0], xticklabels=[],aspect='auto')
        x_plot = fig.add_subplot(grid[-1, 1:], yticklabels=[],aspect='auto')
        main_plot.imshow(RP, cmap='hot')
        main_plot.set_title(title,fontsize='x-small')
        main_plot.set_xticks([])
        main_plot.set_yticks([])
        main_plot.set_xlabel('X(i)',fontsize='large')
        main_plot.set_ylabel('Y(j)',fontsize='large')
        x_plot.set_xmargin(0)
        x_plot.set_ymargin(0)
        y_plot.set_xmargin(0)
        y_plot.set_ymargin(0)
        if TYPE in 'RQA' or TYPE in 'MDRQA':
            if DATA.ndim == 1: DATA = np.expand_dims(DATA,axis=0)
            x_plot.plot(np.arange(DATA.shape[1]),DATA[0],'-k')
            y_plot.plot(np.flip(DATA[0]),np.arange(DATA.shape[1],0,-1),'-k')
        elif TYPE in 'CRQA':
            x_plot.plot(np.arange(DATA.shape[1]),DATA[0,:], 'k-')
            y_plot.plot(np.flip(DATA[1,:]),np.arange(DATA.shape[1],0,-1),'-k') #HACK: Might be plotting backward.
        elif TYPE in 'JRQA':
            # TODO: MATLAB side shows only one line along the x-axis, our plot ends up plotting more than one line... which one is correct?
            y_plot.plot(np.flip(DATA[0,:]),np.arange(DATA.shape[1]),'-k')
            y_plot.invert_yaxis()
            for i in range(r):
                x_plot.plot(np.arange(DATA.shape[1]),DATA[i,:], 'k-')     
        fig.text(.02,.24,"EntrW = {:.2f}".format(RESULTS["EntrW"]))
        plt.show()
    return out


# Function for setting radius to achieve a certain recurrence
def setRadius(a, TYPE, radius, SETVALUE, radiusEnd, LINELENGTH):
    """
    Usage: (perRec, radiusFinal) = setRadius(a, TYPE, radius, SETVALUE, radiusEnd, LINELENGTH)
    Input:  - a, pairwise euclidean distance values of our DATA matrix.
            - TYPE, a string indicating which type of RQA to run (i.e.
                'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
                TYPE = 'RQA'.
            - radius,
            - SETVALUE,sets the value of the selected parameter. If
                SETVALUE = 1, then the radius will be set to 1 if SETPARA
                = 'radius' or the radius will be adjusted until the
                recurrence is equal to 1 if SETPARA = 'recurrence'. The
                default value if SETPARA = 'radius' is 1. The default
                value if SETPARA = 'recurrence' is 2.5.
            - radiusEnd,
            - LINELENGTH,
    Output: - perRec,
            - radiusFinal,
    """
    # Find the radius to provide target # recurrence
    perRec = rqaPerRec(a.copy(), TYPE, radius, LINELENGTH)
    while perRec == 0 or perRec > 2.5:
        # if radius is too small
        #     print('Minimum radius has been adjusted...')
        #     radiusEnd = radius + 0.5;
        if perRec == 0:
            radius = radius*2
        elif perRec > SETVALUE:
            radius = radius / 1.5
            #                 radiusEnd =  radius + 0.5
        perRec = rqaPerRec(a.copy(), TYPE, radius, LINELENGTH)


    perRec = rqaPerRec(a.copy(), TYPE, radiusEnd, LINELENGTH)
    while perRec < SETVALUE:
        # if radiusEnd is too large
        # print('Maximum radius has been increased...')
        radiusEnd = radiusEnd*2
        perRec = rqaPerRec(a.copy(), TYPE, radiusEnd, LINELENGTH)


    # Search for radius with target # recurrence

    target = SETVALUE  # designate what percent recurrence is wanted
    iterations = 20 # Number of iterations to find radius
    lv = np.zeros(iterations+1) # +1 to hold initial low value
    hv = np.zeros(iterations+1) # +1 to hold initial high value
    lv[0] = radius    # set low value
    hv[0] = radiusEnd # set high value
    perRecIter = np.zeros(iterations)
    mid = np.zeros(iterations)
    rad = np.zeros(iterations)
    #TODO: This converges to a different value than the MATLAB code.
    for i1 in range(iterations):
        mid[i1] = (lv[i1]+hv[i1])/2  # find midpoint between hv and lv
        rad[i1] = mid[i1]    # new radius for this iteration
        #Compute recurrence matrix
        perRec = rqaPerRec(a.copy(), TYPE, rad[i1], LINELENGTH)
        
        perRecIter[i1] = perRec  # set percent recurrence
        
        if perRecIter[i1] < target:
            # if percent recurrence is below target percent recurrence
            hv[i1+1] = hv[i1]
            lv[i1+1] = mid[i1]
        else:
            # if percent recurrence is above or equal to target percent recurrence
            lv[i1+1] = lv[i1]
            hv[i1+1] = mid[i1]

    perRecFinal = perRecIter[-1]  # set final percent recurrence
    radiusFinal = rad[-1]      # set radius for final percent recurrence

    return perRecFinal, radiusFinal

def rqaPerRec(a, TYPE, radius, LINELENGTH):
    """
    Usage: perRec = rqaPerRec(A, TYPE, radius, LINELENGTH)
    Input:  - a, pairwise euclidean distance values from our DATA matrix.
            - TYPE, a string indicating which type of RQA to run (i.e.
               'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
               TYPE = 'RQA'.
            - radius, 
            - LINELENGTH
    Output: - perRec
    """
    
    if not isinstance(a,list):
        a = [a]
    for i2 in range(len(a)):
        nradius = np.negative(radius)
        a[i2][np.where(a[i2] >= nradius)] = 1
        a[i2][np.where(a[i2] < nradius)] = 0
    if len(a) > LINELENGTH:
        for i3 in range(len(a)-1):
            a[i3+1] = np.multiply(a[i3],a[i3+1])
        a = a[i3+1]
    else:
        a = a[0]

    # Calculate percent recurrence
    if 'CRQA' not in TYPE:
        perRec = 100*(np.sum(a)-len(a))/(len(a)**2-len(a))
    else:
        perRec = 100*(np.sum(a))/(len(a)**2)
    
    return perRec

def rqaHistograms(a, DATA, TYPE, radius):
    """
    Usage: (diag_hist, vertical_hist) = rqaHistograms(A, DATA, TYPE, radius)
    Input:  - a, stores the recurrence matrix.
            - DATA,
            - TYPE,
            - radius
    Output: - diag_hist,
            - vertical_hist,
    """
    if not isinstance(a,list):
        a = [a]

    for i2 in range(len(a)):
        nradius = np.negative(radius)
        a[i2][np.where(a[i2] >= nradius)] = 1
        a[i2][np.where(a[i2] < nradius)] = 0

    if len(a) > 1:
        for i3 in range(len(a)-1):
            a[i3+1] = np.multiply(a[i3],a[i3+1])
        a = a[i3+1]
    else:
        a = a[0]

    # If one dimensional, expand to two dimensional for the work up ahead.
    if DATA.ndim == 1:
        DATA = np.expand_dims(DATA,axis=0)

    diag_hist = np.array([])
    vertical_hist = np.array([])
    for i4 in range(-DATA.shape[1],DATA.shape[1]): # caluculate diagonal line distribution
        diagonal = np.diag(a.copy(), k=i4).astype(int)
        d=label_components(diagonal)
        # lengths
        d = d[np.nonzero(d)]
        d = np.bincount(d)[1:]

        diag_hist = np.append(diag_hist, d)

    # This removes the line of identity in RQA, jRQA, and mdRQA
    if 'CRQA' not in TYPE:
        diag_hist = diag_hist[diag_hist < np.max(diag_hist)]
        if len(diag_hist) == 0:
            diag_hist = 0
    
    for i5 in range(DATA.shape[1]):
        C = a[:,i5].copy().astype(int)
        v = label_components(C)

        v = v[np.nonzero(v)]
        v = np.bincount(v)[1:]

        vertical_hist = np.append(vertical_hist, v)
    
    return (diag_hist, vertical_hist)
    
# Calculate entropy of weighted recurrence plot
def RQA_WeightedEntropy(WRP):
    """
    Usage: Swrp = RQA_WeightedEntropy(WRP)
    Input:  WRP,
    Output: Swrp,
    """
    N = len(WRP)
    si = np.zeros(N)
    for j in range(N):
        si[j] = np.sum(WRP[:,j])
    mi = min(si)
    ma = max(si)
    m = (ma - mi)/49
    I = 1
    S = np.sum(si)
    p1 = np.array([])
    step = m

    # append to p1 the initial value
    P = np.sum(si[(si >= mi) & (si < (mi+step))])
    p1 = np.append(p1, P/S)
    m = m + mi
    while m < ma:
        # sum of values within the range between m and m+step
        P = np.sum(si[(si >= m) & (si < (m+step))])
        p1 = np.append(p1, P/S)
        m += step
    pp = np.zeros(len(p1))
    for i in range(len(p1)):
        pp[i] = p1[i]*np.log(p1[i])

    pp[np.isnan(pp)] = 0
    
    Swrp = -1*(np.sum(pp))
    return Swrp


def label_components(bimg):
    """
    Input:  - bimg: The binary column to process
    Output: - L: The labeled array.
    Remarks:
        This subfunction serves to model the bwlabel function that exists in the MATLAB image processing library.
        Link: https://www.mathworks.com/help/images/ref/bwlabel.html

        The only exception is that this subfunction only handles one dimensional input, we are only expecting
        similar input from what you see in the RQA subfunction rqaHistograms. Our input consists of a NumPy
        array that may look something like: [0 1 1 1 0 0 0 1 1 0 1], our output would look something like
        [0 1 1 1 0 0 0 2 2 0 3], which gives each 'component' it's own label starting from 1 and incrementing
        for each time we encounter a 1 after encountering 0.

        The function in MATLAB takes in a secondary function which outlines connectivity, due to the input being
        one dimensional, we're ignoring that.
    """
    bimg.setflags(write=1)
    label_value = 0
    in_component = False # using a boolean so we are not incrementing while we are within a certain 'component'
    for i in range(bimg.shape[0]):
        if bimg[i] == 1 and not in_component:
            # mark that we are in a component, increment component value, assign that value to our 1s.
            in_component = True
            label_value += 1
            bimg[i] = label_value
        elif bimg[i] == 1 and in_component:
            # while we are in a component, just mark the values with the component value
            bimg[i] = label_value
        else:
            # if we encounter a 0, then we are not in_component.
            in_component = False
    return bimg


