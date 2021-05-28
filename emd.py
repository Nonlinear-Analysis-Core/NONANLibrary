from typing import Any
from format_processor import format_processor
import sys, warnings
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

#EMD  computes Empirical Mode Decomposition
#
#
#   Syntax
#
#
# IMF = EMD(X)
# IMF = EMD(X,...,'Option_name',Option_value,...)
# IMF = EMD(X,OPTS)
# [IMF,ORT,NB_ITERATIONS] = EMD(...)
#
#
#   Description
#
#
# IMF = EMD(X) where X is a real vector computes the Empirical Mode
# Decomposition [1] of X, resulting in a matrix IMF containing 1 IMF per row, the
# last one being the residue. The default stopping criterion is the one proposed
# in [2]:
#
#   at each point, mean_amplitude < THRESHOLD2*envelope_amplitude
#   &
#   mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE
#   &
#   |#zeros-#extrema|<=1
#
# where mean_amplitude = abs(envelope_max+envelope_min)/2
# and envelope_amplitude = abs(envelope_max-envelope_min)/2
# 
# IMF = EMD(X) where X is a complex vector computes Bivariate Empirical Mode
# Decomposition [3] of X, resulting in a matrix IMF containing 1 IMF per row, the
# last one being the residue. The default stopping criterion is similar to the
# one proposed in [2]:
#
#   at each point, mean_amplitude < THRESHOLD2*envelope_amplitude
#   &
#   mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE
#
# where mean_amplitude and envelope_amplitude have definitions similar to the
# real case
#
# IMF = EMD(X,...,'Option_name',Option_value,...) sets options Option_name to
# the specified Option_value (see Options)
#
# IMF = EMD(X,OPTS) is equivalent to the above syntax provided OPTS is a struct 
# object with field names corresponding to option names and field values being the 
# associated values 
#
# [IMF,ORT,NB_ITERATIONS] = EMD(...) returns an index of orthogonality
#                       ________
#         _  |IMF(i,:).*IMF(j,:)|
#   ORT = \ _____________________
#         /
#         ¯        || X ||²
#        i~=j
#
# and the number of iterations to extract each mode in NB_ITERATIONS
#
#
#   Options
#
#
#  stopping criterion options:
#
# STOP: vector of stopping parameters [THRESHOLD,THRESHOLD2,TOLERANCE]
# if the input vector's length is less than 3, only the first parameters are
# set, the remaining ones taking default values.
# default: [0.05,0.5,0.05]
#
# FIX (int): disable the default stopping criterion and do exactly <FIX> 
# number of sifting iterations for each mode
#
# FIX_H (int): disable the default stopping criterion and do <FIX_H> sifting 
# iterations with |#zeros-#extrema|<=1 to stop [4]
#
#  bivariate/complex EMD options:
#
# COMPLEX_VERSION: selects the algorithm used for complex EMD ([3])
# COMPLEX_VERSION = 1: "algorithm 1"
# COMPLEX_VERSION = 2: "algorithm 2" (default)
# 
# NDIRS: number of directions in which envelopes are computed (default 4)
# rem: the actual number of directions (according to [3]) is 2*NDIRS
# 
#  other options:
#
# T: sampling times (line vector) (default: 1:length(x))
#
# MAXITERATIONS: maximum number of sifting iterations for the computation of each
# mode (default: 2000)
#
# MAXMODES: maximum number of imfs extracted (default: Inf)
#
# DISPLAY: if equals to 1 shows sifting steps with pause
# if equals to 2 shows sifting steps without pause (movie style)
# rem: display is disabled when the input is complex
#
# INTERP: interpolation scheme: 'linear', 'cubic', 'pchip' or 'spline' (default)
# see numpy.interp1d documentation for details
#
# MASK: masking signal used to improve the decomposition according to [5]
#
#
#   Examples
#
#
#X = rand(1,512);
#
#IMF = emd(X);
#
#IMF = emd(X,'STOP',[0.1,0.5,0.05],'MAXITERATIONS',100);
#
#T=linspace(0,20,1e3);
#X = 2*exp(i*T)+exp(3*i*T)+.5*T;
#IMF = emd(X,'T',T);
#
#OPTIONS.DISLPAY = 1;
#OPTIONS.FIX = 10;
#OPTIONS.MAXMODES = 3;
#[IMF,ORT,NBITS] = emd(X,OPTIONS);
#
#
#   References
#
#
# [1] N. E. Huang et al., "The empirical mode decomposition and the
# Hilbert spectrum for non-linear and non stationary time series analysis",
# Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
#
# [2] G. Rilling, P. Flandrin and P. Gonçalves
# "On Empirical Mode Decomposition and its algorithms",
# IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
# NSIP-03, Grado (I), June 2003
#
# [3] G. Rilling, P. Flandrin, P. Gonçalves and J. M. Lilly.,
# "Bivariate Empirical Mode Decomposition",
# Signal Processing Letters (submitted)
#
# [4] N. E. Huang et al., "A confidence limit for the Empirical Mode
# Decomposition and Hilbert spectral analysis",
# Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003
#
# [5] R. Deering and J. F. Kaiser, "The use of a masking signal to improve 
# empirical mode decomposition", ICASSP 2005
#
#
# See also
#  emd_visu (visualization),
#  emdc, emdc_fix (fast implementations of EMD),
#  cemdc, cemdc_fix, cemdc2, cemdc2_fix (fast implementations of bivariate EMD),
#  hhspectrum (Hilbert-Huang spectrum)
#
#
# G. Rilling, last modification: 3.2007
# gabriel.rilling@ens-lyon.fr

#TODO: Stubbed off plotting function.
def display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting) -> None:
    pass

#TODO: Stubbed off plotting function.
def display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sp,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift) -> None:
    pass

def extr(x : np.ndarray, nargout : int = 2):
    t = np.arange(0,len(x)) # NOTE: Shape may be more accurate to the MATLAB code

    m = len(x)

    if nargout > 2:
        x1 = x[:m-1]      #NOTE: Make sure this is taking in the same number of points, may be off by one
        x2 = x[1:m]
        indzer = np.argwhere(np.multiply(x1,x2) < 0)   # where element-wise product is negative
        if any(x == 0):
            iz = np.where(x==0)
            indz = np.array([])
            if any(np.diff(iz)==1):
                zer = x == 0
                dz = np.diff(np.array([0, zer, 0]))
                debz = np.where(dz == 1)
                finz = np.where(dz == -1) -1

                indz = round((debz+finz)/2)
            else:
                indz = iz
            if 'indzer' in locals():  # NOTE: Errors likely
                indzer = np.sort(np.concatenate(indzer,indz))
            else:
                indzer = np.sort(indz)

    d = np.diff(x)      #NOTE (5/18) : Given the exact same input, the MATLAB code and the Python diff functions return different values.

    n = len(d)
    d1 = d[:n-1]
    d2 = d[1:n+1]
    indmin = np.argwhere((np.multiply(d1,d2) < 0) & (d1 < 0))+1
    indmax = np.argwhere((np.multiply(d1,d2) < 0) & (d1 > 0))+1

    # when two or more successive points have the same value we consider only one extremum in the middle of the constant area
    # (only works if the signal is uniformly sampled)

    if any(d==0):

        imax = np.array([])
        imin = np.array([])

        bad = d==0
        dd = np.diff(np.array([0, bad, 0])) if len(bad) == 1 else np.diff(np.concatenate((np.array([0]),bad,np.array([0]))))
        debs = np.where(dd == 1)
        fins = np.where(dd == -1)
        if debs[0] == 0:
            if len(debs) > 1:
                debs = debs[1:]
                fins = fins[1:]
            else:
                debs = np.array([])
                fins = np.array([])
        if len(debs) > 0:
            if fins[-1] == m:
                if len(debs) > 1:
                    debs = debs[:-1]
                    fins = fins[:-1]
                else:
                    debs = np.array([])
                    fins = np.array([])
        lc = len(debs)
        if lc > 0:
            for k in range(lc):
                if d[debs[k]-1] > 0:
                    if d[fins[k]] < 0:
                        imax = np.append(imax, round((fins[k] + debs[k])/2))
                else:
                    if d[fins[k]] > 0:
                        imin = np.append(imin, round((fins[k] + debs[k])/2))
        if len(imax) > 0:
            indmax = np.sort(np.concatenate((indmax,imax))) # NOTE: bound to have errors

        if len(imin) > 0:
            indmin = np.sort(np.concatenate((indmin,imax))) # NOTE: bound to have errors
    
    if nargout == 2:
        return indmin, indmax
    elif nargout == 3:
        return indmin, indmax, indzer
    else:
        raise ValueError('No output supported for nargout of > 3 or < 2.')

def boundary_conditions(indmin : np.ndarray, indmax : np.ndarray, t : np.ndarray, x : np.ndarray, z : np.ndarray, nbsym : int):
    # returns (tmin, tmax, zmin, zmax)
    lx = len(x)-1  #NOTE: Subtracting one here to reflect the Python 'final' index. May be unneccessary
    indmin = indmin.flatten()
    indmax = indmax.flatten()

    if len(indmin) + len(indmax) < 3:
        raise ValueError('not enough extrema')
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1:min(indmax[-1],nbsym+1)]).flatten()
            lmin = np.flipud(indmin[:min(indmin[-1],nbsym)]).flatten()
            lsym = indmax[0]
        else:
            lmax = np.flipud(indmax[:min(indmax[-1],nbsym)]).flatten()
            lmin = np.array([np.flipud(indmin[:min(indmin[-1],nbsym-1)]),1])
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[:min(indmax[-1],nbsym)]).flatten()
            lmin = np.flipud(indmin[1:min(indmin[-1],nbsym+1)]).flatten()
            lsym = indmin[0]
        else:
            lmax = np.array([np.flipud(indmax[:min(indmax[-1],nbsym-1)]),1])
            lmin = np.flipud(indmin[:min(indmin[-1],nbsym)]).flatten()
            lsym = 0
    
    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[-1-nbsym+1:]) if indmax.size >= nbsym+1 else np.flipud(indmax[:])
            rmin = np.flipud(indmin[-1-nbsym:-1]) if indmin.size >= nbsym   else np.flipud(indmin[:])
            rsym = indmin[-1]
        else:
            rmax = np.concatenate((np.array([lx]), np.flipud(indmax[-1-nbsym+2:]))) if indmax.size >= nbsym+2 else np.concatenate((np.array([lx]),indmax[:]))
            rmin = np.flipud(indmin[-1-nbsym+1:]) if indmin.size >= nbsym+1 else np.flipud(indmin[:])
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[-1-nbsym:-1]) if indmax.size >= nbsym else np.flipud(indmax[:])
            rmin = np.flipud(indmin[-1-nbsym+1:]) if indmax.size >= nbsym+1 else np.flipud(indmin[:])
            rsym = indmax[-1]
        else:
            rmax = np.flipud(indmax[-1-nbsym+1:]) if indmax.size >= nbsym+1 else np.flipud(indmax[:])
            rmin = np.concatenate((np.array([lx]), np.flipud(indmin[-1-nbsym+2:]))) if indmax.size >= nbsym+2 else np.concatenate((np.array([lx]),indmin[:]))
            rsym = lx
    
    tlmin = 2*t[lsym]-t[lmin]
    tlmax = 2*t[lsym]-t[lmax]
    trmin = 2*t[rsym]-t[rmin]
    trmax = 2*t[rsym]-t[rmax]
    
    # in case symmetrized parts do not extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:min(indmax[-1],nbsym)])
        else:
            lmin = np.flipud(indmin[:min(indmin[-1],nbsym)])
        if lsym == 1:
            raise ValueError('bug')
        lsym = 1
        tlmin = 2*t[lsym]-t[lmin]
        tlmax = 2*t[lsym]-t[lmax]
    
    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(indmax[-1]-nbsym+1,1):])
        else:
            rmin = np.flipud(indmin[max(indmin[-1]-nbsym+1,1):])
        if rsym == lx:
            raise ValueError('bug')
        rsym = lx
        trmin = 2*t[rsym]-t[rmin]
        trmax = 2*t[rsym]-t[rmax]
        
    zlmax = z[lmax]
    zlmin = z[lmin]
    zrmax = z[rmax]
    zrmin = z[rmin]
    
    tmin = np.concatenate((tlmin, t[indmin], trmin))
    tmax = np.concatenate((tlmax, t[indmax], trmax))
    zmin = np.concatenate((zlmin, z[indmin], zrmin))
    zmax = np.concatenate((zlmax, z[indmax], zrmax))

    return (tmin, tmax, zmin, zmax)

def stop_EMD(r : np.ndarray, MODE_COMPLEX : int, ndirs : int):
    ner = np.zeros(ndirs)
    if MODE_COMPLEX:
        for k in range(ndirs):
            phi = (k-1)*np.pi/ndirs
            (indmin,indmax) = extr(np.real(np.exp(1.j*phi)*r))
            ner[k] = len(indmin) + len(indmax) #NOTE: See how these two functions interact
        stop = any(ner < 3)
    else:
        (indmin, indmax) = extr(r,nargout=2)
        ner = len(indmin) + len(indmax)
        stop = ner < 3

    return stop

def emd(*args : Any, nargout : int = 1): # NOTE: Returns (imf, ort, nbits)

    (x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask) = init(*args)

    if display_sifting:
        fig_h = plt.figure()


    nbits = np.zeros(MAXMODES)
    #main loop : requires at least 3 extrema to proceed
    while not stop_EMD(r, MODE_COMPLEX, ndirs) and (k < MAXMODES+1 or MAXMODES == 0) and not mask:
        # current mode
        m = r
        # mode at previous iteration
        mp = m
        #computation of mean and stopping criterion
        if FIXE:
            (stop_sift,moyenne) = stop_sifting_fixe(t,m,INTERP,MODE_COMPLEX,ndirs)
        elif FIXE_H:
            stop_count = 0
            (stop_sift,moyenne) = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs)
        else:
            (stop_sift,moyenne) = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs,nargout=2)

        # in case the current mode is so small that machine precision can cause
        # spurious extrema to appear
        if (max(abs(m))) < (1e-10)*(max(abs(x))):
            if not stop_sift:
                warnings.warn('emd:warning forced stop of EMD : too small amplitude')
            else:
                print('forced stop of EMD : too small amplitude')
                break


    # # sifting loop
        while not stop_sift and nbit<MAXITERATIONS:           
            #NOTE: Check output formatting, when getting to this section of the code.
            if not MODE_COMPLEX and nbit>MAXITERATIONS/5 and nbit % np.floor(MAXITERATIONS/10) == 0 and not FIXE and nbit > 100:
                print('mode', k, ', iteration', nbit)
                if 's' in locals():
                    print('stop parameter mean value:',s)
                (im,iM) = extr(m,nargout=2)
                print(sum(m[im] > 0),'minima > 0;', sum(m[iM] < 0),'maxima < 0.')
            #sifting
            m = m - moyenne
            #computation of mean and stopping criterion
            if FIXE:
                (stop_sift,moyenne) = stop_sifting_fixe(t,m,INTERP,MODE_COMPLEX,ndirs)
            elif FIXE_H:
                (stop_sift,moyenne,stop_count) = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs)
            else:
                (stop_sift,moyenne,s) = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs,nargout=3)

            # display
            if display_sifting and not MODE_COMPLEX:
                NBSYM = 2
                (indmin,indmax) = extr(mp, nargout=2)
                (tmin,tmax,mmin,mmax) = boundary_conditions(indmin,indmax,t,mp,mp,NBSYM)
                envminp = interp.interp1d(tmin, mmin, kind=INTERP)
                envmaxp = interp.interp1d(tmax,mmax,kind=INTERP)
                envmoyp = (envminp+envmaxp)/2
                if FIXE or FIXE_H:
                    display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting)       #TODO: stub off this, plotting function
                else:
                    sxp=2*np.divide(abs(envmoyp),abs(envmaxp-envminp))
                    sp = np.mean(sxp)
                    display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sp,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift)    #TODO: ditto above

            mp = m
            nbit=nbit+1
            NbIt=NbIt+1

            if nbit == (MAXITERATIONS-1) and not FIXE and nbit > 100:
                if s in vars():
                    # TODO: These warnings are a little different from the MATLAB code, it may show differing output.
                    warnings.warn('emd:warning forced stop of sifting : too many iterations... mode',k,'. stop parameter mean value : ',s)
                else:
                    warnings.warn('emd:warning forced stop of sifting : too many iterations... mode',k,'.')
        # sifting loop
        imf = np.vstack((imf,m)) if imf.size != 0 else np.append(imf,m)

        if display_sifting:
            print('mode', k,'stored')
        nbits = np.append(nbits,nbit)
        k = k+1

        r = r - m
        nbit=0
        #end main loop
    if any(r) and not mask:
        imf = np.vstack((imf,r)) if imf.size != 0 else np.append(imf,r)
        
    ort = io(x,imf)
    
    if nargout == 1:
        return imf
    elif nargout == 3:
        return imf, ort, nbits
    else:
        raise ValueError('Errored on value of "nargout" output arguments.\
        \nValid options are "nargout=1" and "nargout=3".\
        \nCheck documentation for additional details.')

# computes the mean of the envelopes and the mode amplitude estimate
def mean_and_amplitude(m : np.ndarray, t : np.ndarray, INTERP : str, MODE_COMPLEX : int, ndirs : int, nargout : int = 4):      # returns (envmoy, nem, nzm, amp), 
    #TODO: fix the nargout to adapt to the multiple usages in this file
    NBSYM = 2
    nem = np.zeros(ndirs)
    nzm = np.zeros(ndirs)
    envmin = np.zeros(ndirs)
    envmax = np.zeros(ndirs)

    if MODE_COMPLEX:
        if MODE_COMPLEX == 1:
            # do something
            for k in range(ndirs):
                phi = (k-1)*np.pi/ndirs
                y = np.real(np.exp(-1.j*phi)*m)
                indmin, indmax, indzer = extr(y, nargout=3) # TODO: Make sure the nargout in the extr function is working.
                nem[k] = len(indmin) + len(indmax)
                nzm[k] = len(indzer)
                tmin, tmax, zmin, zmax = boundary_conditions(indmin, indmax, t, y, m, NBSYM)
                envmin[k] = interp.interp1d(tmin,zmin,kind=INTERP)  # NOTE: Undoubtedly this has bugs, but the MATLAB version is vague here
                envmax[k] = interp.interp1d(tmax,zmax,kind=INTERP)
            envmoy = np.mean((envmin-envmax)/2,axis=0)
            if nargout > 3:
                amp = np.mean(np.abs(envmax-envmin),axis=0)/2
        elif MODE_COMPLEX == 2:
            for k in range(ndirs):
                phi = (k-1)*np.pi/ndirs
                y = np.real(np.exp(-1.j*phi)*m)
                indmin, indmax, indzer = extr(y,nargout=3) # TODO: Make sure the nargout in the extr function is working.
                nem[k] = len(indmin) + len(indmax)
                nzm[k] = len(indzer)
                (tmin, tmax, zmin, zmax) = boundary_conditions(indmin, indmax, t,y,y,NBSYM) # NOTE: y is input twice?
                envmin[k] = np.exp(1.j*phi)*interp.interp1d(tmin,zmin,kind=INTERP)
                envmax[k] = np.exp(1.j*phi)*interp.interp1d(tmax,zmax,kind=INTERP)
            envmoy = np.mean((envmin+envmax),axis=0)
            if nargout > 3:
                amp = np.mean(np.abs(envmax-envmin),axis=0)/2
    else:
        (indmin, indmax, indzer) = extr(m, nargout=3)
        nem = len(indmin) + len(indmax)
        nzm = len(indzer)
        (tmin, tmax, mmin, mmax) = boundary_conditions(indmin, indmax, t, m, m, NBSYM)
        if 'linear' in INTERP:  #NOTE: This is the answer for a linear interpolation option.
            envmin = np.interp(t, tmin, mmin)
        # elif 'spline' in INTERP: #NOTE: This does not yet work correctly as an option for spline interpolation.
        #     tck = interp.splrep(tmin, mmin,)
        #     xnew = np.arange(0, 2*np.pi, np.pi/50)
        #     ynew = interp.splev(xnew, tck, der=0)
        #     envmin = interp.interp1d(tmin,mmin,kind=INTERP)
        if 'linear' in INTERP:
            envmax = np.interp(t,tmax,mmax)
        # elif 'spline' in INTERP:
        #     envmax = interp.interp1d(tmax,mmax,kind=INTERP)
        envmoy = (envmin+envmax)/2
        if nargout > 3:
            # expand dims needed for mean function.
            amp = np.mean(np.expand_dims(np.abs(envmax-envmin),0),axis=0)/2
    
    if nargout == 4:
        return (envmoy,nem,nzm,amp)
    elif nargout == 3:
        return (envmoy,nem,nzm)
    elif nargout == 1:
        return envmoy
    else:
        raise ValueError('Errored on value of "nargout" output arguments.\
        \nValid options are "nargout=4","nargout=3", and "nargout=1".\
        \nCheck documentation for additional details.')   

def stop_sifting(m:np.ndarray,t:np.ndarray,sd:float,sd2:float,tol:float,INTERP:str,MODE_COMPLEX:int,ndirs:int,nargout:int=2):
    try:
        (envmoy,nem,nzm,amp) = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs,nargout=4)
        sx = np.divide(abs(envmoy),amp)
        s = np.mean(sx)
        stop = not ((np.mean(sx > sd) > tol or (sx > sd2)) and (nem > 2))     # NOTE: This may work correctly.
        if not MODE_COMPLEX:
            stop = stop and not (abs(nzm-nem) > 1)
    except:
        stop = 1
        envmoy = np.zeros((1, len(m)))
        s = np.nan
    finally:
        if nargout == 2:
            return stop, envmoy
        elif nargout == 3:
            return stop, envmoy, s
        else:
            raise ValueError('Errored on value of "nargout" output arguments.\
        \nValid options are "nargout=3" and "nargout=2".\
        \nCheck documentation for additional details.') 

def stop_sifting_fixe(t:np.ndarray,m:np.ndarray,INTERP:str,MODE_COMPLEX:int,ndirs:int):       # return (stop,moyenne)
    try:
        moyenne = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs,nargout=1)
        stop = 0
    except:
        moyenne = np.zeros((1,len(m)))
        stop = 1
    finally:
        return stop,moyenne # NOTE: might not be a good idea to have this in a finally block, look at this later.

def stop_sifting_fixe_h(t:np.ndarray,m:np.ndarray,INTERP:str,stop_count:int,FIXE_H:int,MODE_COMPLEX:int,ndirs:int,nargout:int=2):
    try:
        (moyenne,nem,nzm) = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs,nargout=3)
        if all(abs(nzm-nem)>1):
            stop = 0
            stop_count = 0
        else:
            stop_count = stop_count+1
            stop = (stop_count == FIXE_H)
    except:
        moyenne = np.zeros((1,len(m))) 
        stop = 1
        stop_count = 0 # NOTE: This line is added, not sure if necessary.
    finally:
        if nargout == 2:
            return stop,moyenne
        elif nargout == 3:
            return stop,moyenne,stop_count
        else:
            raise ValueError('Errored on value of "nargout" output arguments.\
            \nValid options are "nargout=2" and "nargout=3".\
            \nCheck documentation for additional details.')

def io(x : np.ndarray, imf : np.ndarray):
    """
     ort = io(x,imf) computes the index of orthogonality
    
    inputs : - x    : analyzed signal
             - imf  : empirical mode decomposition
    """
    n = imf.shape[0]
    s = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                s = s + abs(np.sum(np.multiply(imf[i],np.conjugate(imf[j]))/sum(np.power(x,2))))

    return 0.5*s # our 'ort' value

# Returns x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask
def init(*argv : Any):
    if not isinstance(argv[0], np.ndarray):
        x = np.array(argv[0])
    if len(argv) == 2:
        if isinstance(argv[1], dict):
            inopts = argv[1]
        else:
            raise ValueError('when using 2 arguments the first one is the analyzed signal X and the second one is a dictionary object describing the options')
    elif len(argv) > 2:
        try:
            inopts = list(argv[1:]) # using a list until later when we build the dictionary more.
        except:
            raise Exception('bad argument syntax')

    # default for stopping
    defstop = np.array([0.05,0.5,0.05])

    opt_fields = {'t','stop','display','maxiterations','fix','maxmodes','interp','fix_h','mask','ndirs','complex_version'}

    defopts = dict()
    defopts["stop"] = defstop
    defopts["display"] = 0
    defopts["t"] = np.arange(0,max(x.shape)) 
    defopts["maxiterations"] = 2000
    defopts["fix"] = 0
    defopts["maxmodes"] = 0
    defopts["interp"] = 'linear'
    defopts["fix_h"] = 0
    defopts["mask"] = 0
    defopts["ndirs"] = 4
    defopts["complex_version"] = 2

    opts = defopts



    if len(argv) == 1:
        inopts = defopts
    elif len(argv) == 0:
        raise TypeError('Not enough arguments')

    names = inopts.keys()
    for name in names:
        if name not in opt_fields:
            raise NameError('bad option field name, check capitalization: ',name)
        if name not in inopts.keys():    # discard empties
            opts[name] = inopts[name]

    t = opts["t"]
    stop = opts["stop"]
    display_sifting = opts["display"]
    MAXITERATIONS = opts["maxiterations"]
    FIXE = opts["fix"]
    MAXMODES = opts["maxmodes"]
    INTERP = opts["interp"]
    FIXE_H = opts["fix_h"]
    mask = opts["mask"]
    ndirs = opts["ndirs"]
    complex_version = opts["complex_version"]

    # NOTE: In the future, might accept two dimensional input where the shape is defined by (1,n) or (n,1)
    if x.ndim != 1:
        raise TypeError('X must have only one row or one column.')

    # NOTE: This checking does not matter as the previous if only allows one dimensional vector inputs
    # if size(x,1) > 1
    #     x = x.';
    # end

    if t.ndim != 1:
        raise TypeError('Option field T must have only one row or one column')

    if not np.isreal(t.all()):
        raise TypeError('Time instants T must be a real vector')

    # NOTE: Ditto from last note.
    # if size(t,1) > 1
    #     t = t';
    # end

    if t.shape[0] != x.shape[0]:
        raise TypeError('X and option field T must have the same length')

    if stop.ndim != 1 or stop.shape[0] > 3:
        raise TypeError('Option field STOP must have only one row or one column of max three elements')

    if not all(np.isfinite(x)):
        raise ValueError('Data elements must be finite')

    # if size(stop,1) > 1
    #     stop = stop';
    # end

    L = len(stop)
    if L < 3:
        stop[2] = defstop[2]
    if L < 2:
        stop[1] = defstop[1]

    if not isinstance(INTERP,str) or INTERP not in {'linear', 'cubic', 'spline'}:
        raise TypeError('INTERP field must be linear, cubic, or spline')

    #special procedure when a masking signal is specified
    # NOTE: Keeping to the MATLAB code for now on the any(mask) portion, may change in the future.
    if mask:
        if mask.ndim == 1 or mask.shape[0] != x.shape[0]:
            raise TypeError('Masking signal must have the same dimension as the analyzed signal X')
        

        # if size(mask,1) > 1
        #     mask = mask.';
        # end
        opts["mask"] = 0
        imf1 = emd(x+mask,opts)
        imf2 = emd(x-mask,opts)
        if imf1.shape[0] != imf2.shape[0]:
            raise Warning('emd:warning, the two sets of IMFs have different sizes:', imf1, 'and', imf2, 'IMFs.')
        S1 = imf1.shape[0]
        S2 = imf2.shape[0]
        if S1 != S2:
            if S1 < S2:
                tmp = imf1
                imf1 = imf2
                imf2 = tmp
            imf2[max(S1,S2)] = 0
        imf = (imf1+imf2)/2

    sd = stop[0]
    sd2 = stop[1]
    tol = stop[2]

    lx = x.shape[0]

    sdt = sd*np.ones((1,lx))
    sdt = sd*np.ones(lx)
    sd2t = sd2*np.ones((1,lx))

    if FIXE:
        MAXITERATIONS = FIXE
        if FIXE_H:
            raise TypeError('cannot use both ''FIX'' and ''FIX_H'' modes')

    # if not real values found, use complex_version, false otherwise.
    MODE_COMPLEX = not all(np.isreal(x)) * complex_version
    if MODE_COMPLEX and complex_version != 1 and complex_version != 2:
        raise ValueError('COMPLEX_VERSION parameter must be equal to 1 or 2')

    # number of extrema and zero-crossings in residual
    ner = lx
    nzr = lx

    r = x

    if not mask: # if a masking signal is specified "imf" already exists at this stage
        imf = np.array([])
    k = 1

    # iterations counter for extraction of 1 mode
    nbit=0

    # total iterations counter
    NbIt=0
    return (x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask)

if __name__ == '__main__':
    data_input = format_processor()
    emd = emd(data_input[0][1:201],nargout=1)
    print('heree')