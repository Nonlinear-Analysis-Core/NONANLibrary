import matplotlib.pyplot as plt
import warnings
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

def emd(x, t = 0, stop = np.array([0.05,0.5,0.05]), ndirs = 4, display_sifting = 0, MODE_COMPLEX = 2, MAXITERATIONS = 2000, 
FIXE = 0, FIXE_H = 0, MAXMODES = 0, INTERP = 'cubic', mask = 0, nargout = 1):
    """
    EMD  computes Empirical Mode Decomposition


    Syntax


    IMF = EMD(X)
    IMF = EMD(X,...,'Option_name',Option_value,...)
    IMF = EMD(X,OPTS)
    [IMF,ORT,NB_ITERATIONS] = EMD(...)


    Description


    IMF = EMD(X) where X is a real vector computes the Empirical Mode
    Decomposition [1] of X, resulting in a matrix IMF containing 1 IMF per row, the
    last one being the residue. The default stopping criterion is the one proposed
    in [2]:

    at each point, mean_amplitude < THRESHOLD2*envelope_amplitude
    &
    mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE
    &
    |#zeros-#extrema|<=1

    where mean_amplitude = abs(envelope_max+envelope_min)/2
    and envelope_amplitude = abs(envelope_max-envelope_min)/2

    IMF = EMD(X) where X is a complex vector computes Bivariate Empirical Mode
    Decomposition [3] of X, resulting in a matrix IMF containing 1 IMF per row, the
    last one being the residue. The default stopping criterion is similar to the
    one proposed in [2]:

    at each point, mean_amplitude < THRESHOLD2*envelope_amplitude
    &
    mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE

    where mean_amplitude and envelope_amplitude have definitions similar to the
    real case

    IMF = EMD(X,...,'Option_name',Option_value,...) sets options Option_name to
    the specified Option_value (see Options)

    IMF = EMD(X,OPTS) is equivalent to the above syntax provided OPTS is a struct 
    object with field names corresponding to option names and field values being the 
    associated values 

    [IMF,ORT,NB_ITERATIONS] = EMD(...) returns an index of orthogonality
                        ________
            _  |IMF(i,:).*IMF(j,:)|
    ORT = \ _____________________
            /
            ¯        || X ||²
        i~=j

    and the number of iterations to extract each mode in NB_ITERATIONS


    Options


    stopping criterion options:

    STOP: vector of stopping parameters [THRESHOLD,THRESHOLD2,TOLERANCE]
    if the input vector's length is less than 3, only the first parameters are
    set, the remaining ones taking default values.
    default: [0.05,0.5,0.05]

    FIX (int): disable the default stopping criterion and do exactly <FIX> 
    number of sifting iterations for each mode

    FIX_H (int): disable the default stopping criterion and do <FIX_H> sifting 
    iterations with |#zeros-#extrema|<=1 to stop [4]

    bivariate/complex EMD options:

    COMPLEX_VERSION: selects the algorithm used for complex EMD ([3])
    COMPLEX_VERSION = 1: "algorithm 1"
    COMPLEX_VERSION = 2: "algorithm 2" (default)

    NDIRS: number of directions in which envelopes are computed (default 4)
    rem: the actual number of directions (according to [3]) is 2*NDIRS

    other options:

    T: sampling times (line vector) (default: 1:length(x))

    MAXITERATIONS: maximum number of sifting iterations for the computation of each
    mode (default: 2000)

    MAXMODES: maximum number of imfs extracted (default: Inf)

    DISPLAY: if equals to 1 shows sifting steps with pause
    if equals to 2 shows sifting steps without pause (movie style)
    rem: display is disabled when the input is complex

    INTERP: interpolation scheme: 'linear', 'cubic', or 'spline' (default)
    Important to note, 'spline' defaults to using quadratic spline interpolation. For 'cubic' spline interpolation use 'cubic'.
    see numpy.interp1d documentation for details

    MASK: masking signal used to improve the decomposition according to [5]

    References

    [1] N. E. Huang et al., "The empirical mode decomposition and the
    Hilbert spectrum for non-linear and non stationary time series analysis",
    Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998

    [2] G. Rilling, P. Flandrin and P. Gonçalves
    "On Empirical Mode Decomposition and its algorithms",
    IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
    NSIP-03, Grado (I), June 2003

    [3] G. Rilling, P. Flandrin, P. Gonçalves and J. M. Lilly.,
    "Bivariate Empirical Mode Decomposition",
    Signal Processing Letters (submitted)

    [4] N. E. Huang et al., "A confidence limit for the Empirical Mode
    Decomposition and Hilbert spectral analysis",
    Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003

    [5] R. Deering and J. F. Kaiser, "The use of a masking signal to improve 
    empirical mode decomposition", ICASSP 2005

    G. Rilling, last modification: 3.2007
    gabriel.rilling@ens-lyon.fr
    """

    # Initialize variables
    (x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask) \
    = init(x,t,stop,ndirs,display_sifting,MODE_COMPLEX,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask)

    v = 0

    nbits = np.zeros(MAXMODES)
    #main loop : requires at least 3 extrema to proceed
    while not stop_EMD(r, MODE_COMPLEX, ndirs) and (k < MAXMODES+1 or MAXMODES == 0) and not isinstance(mask,np.ndarray):
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
            v+=1           
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
                (stop_sift,moyenne,stop_count) = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs,nargout=3)
            else:
                (stop_sift,moyenne,s) = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs,nargout=3)

            # display
            if display_sifting and not MODE_COMPLEX:
                NBSYM = 2
                (indmin,indmax) = extr(mp, nargout=2)
                (tmin,tmax,mmin,mmax) = boundary_conditions(indmin,indmax,t,mp,mp,NBSYM)
                if 'linear' in INTERP:
                    envminp = np.interp(t,tmin,mmin)
                    envmaxp = np.interp(t,tmax,mmax)
                elif 'quadratic' or 'cubic' or 'spline' in INTERP:
                    f = interp.interp1d(tmin,mmin,kind=INTERP)
                    envminp = f(t)
                    f = interp.interp1d(tmax,mmax,kind=INTERP)
                    envmaxp = f(t)
                envmoyp = (envminp+envmaxp)/2
                if FIXE or FIXE_H:
                    display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting)      
                else:
                    sxp=2*np.divide(abs(envmoyp),abs(envmaxp-envminp))
                    sp = np.mean(sxp)
                    display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sp,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift)   

            mp = m
            nbit=nbit+1
            NbIt=NbIt+1

            if nbit == (MAXITERATIONS-1) and not FIXE and nbit > 100:
                if s in vars():
                    warnings.warn('emd:warning forced stop of sifting : too many iterations... mode {}. stop parameter mean value : {:.4f}'.format(k, s), RuntimeWarning)
                else:
                    warnings.warn('emd:warning forced stop of sifting : too many iterations... mode {} .'.format(k), RuntimeWarning)
        # sifting loop
        imf = np.vstack((imf,m)) if imf.size != 0 else np.append(imf,m)

        if display_sifting:
            print('mode', k,'stored')
        nbits = np.append(nbits,nbit)
        k = k+1

        r = r - m
        nbit=0
        #end main loop
    if any(r) and not isinstance(mask,np.ndarray):
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

def display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting) -> None:
    pass
    fig = plt.figure(figsize=(5,5))
    grid = plt.GridSpec(3,3,hspace=0.7,wspace=0.5)
    plot1 = fig.add_subplot(grid[0,:])
    plot1.plot(t,mp)
    plot1.plot(t,envmaxp,'--k')
    plot1.plot(t,envminp,'--k')
    plot1.plot(t,envmoyp,'r')
    plot1.set_xticks([])
    plot1.set_title('IMF {}; iteration {} before sifting'.format(k,nbit))
    plot2 = fig.add_subplot(grid[1,:])
    plot2.plot(t,m)
    plot2.set_xticks([])
    plot2.set_title('IMF {}; iteration {} after sifting'.format(k,nbit))
    plot3 = fig.add_subplot(grid[2,:])
    plot3.plot(t,r-m)
    plot3.set_title('residue')
    plt.show()

def display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sb,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift):
    fig = plt.figure(figsize=(5,5))
    grid = plt.GridSpec(4,4,hspace=0.7,wspace=0.5)
    plot1 = fig.add_subplot(grid[0,:])
    plot1.plot(t,mp)
    plot1.plot(t,envmaxp,'--k') 
    plot1.plot(t,envminp,'--k')
    plot1.plot(t,envmoyp,'r')
    plot1.set_xticks([])
    plot1.set_title('IMF {};  iteration {} before sifting'.format(k,nbit))
    plot2 = fig.add_subplot(grid[1,:])
    plot2.plot(t,sxp)
    plot2.plot(t,sdt,'--r')
    plot2.plot(t,sd2t,':k')
    plot2.set_title('stop parameter')
    plot2.set_xticks([])
    plot3 = fig.add_subplot(grid[2,:])
    plot3.plot(t,m)
    plot3.set_title('IMF {};  iteration {} after sifting'.format(k,nbit))
    plot3.set_xticks([])
    plot4 = fig.add_subplot(grid[3,:])
    plot4.plot(t,r-m)
    plot4.set_title('residue')
    print('stop parameter mean value : {:.4f} before sifting and {:.4f} after'.format(sb,s))
    plt.show()
    
def extr(x:np.ndarray,nargout:int=2):
    t = np.arange(0,len(x)) 

    m = len(x)

    if nargout > 2:
        x1 = x[:m-1]      
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

    d = np.diff(x)     

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
        debs = np.where(dd == 1)[0]
        fins = np.where(dd == -1)[0]
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
            indmax = np.sort(np.concatenate((indmax.flatten(),imax.flatten())))
            if indmax.dtype == 'float64':
                indmax = indmax.astype('int32')

        if len(imin) > 0:
            indmin = np.sort(np.concatenate((indmin.flatten(),imax.flatten())))
            if indmin.dtype == 'float64':
                indmin = indmin.astype('int32')
    
    if nargout == 2:
        return indmin, indmax
    elif nargout == 3:
        return indmin, indmax, indzer
    else:
        raise ValueError('No output supported for nargout of > 3 or < 2.')

def boundary_conditions(indmin : np.ndarray, indmax : np.ndarray, t : np.ndarray, x : np.ndarray, z : np.ndarray, nbsym : int):
    # returns (tmin, tmax, zmin, zmax)
    lx = len(x)-1  
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
            lmin = np.concatenate((np.flipud(indmin[:min(indmin[-1],nbsym-1)]),np.array([0])))
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[:min(indmax[-1],nbsym)]).flatten()
            lmin = np.flipud(indmin[1:min(indmin[-1],nbsym+1)]).flatten()
            lsym = indmin[0]
        else:
            lmax = np.concatenate((np.flipud(indmax[:min(indmax[-1],nbsym-1)]),np.array([0])))
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
    #TODO: This section needs to be tested.
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
            rmax = np.flipud(indmax[-1-nbsym+1:]) if indmax.size >= nbsym+1 else np.flipud(indmax[:])
        else:
            rmin = np.flipud(indmin[-1-nbsym+1:]) if indmin.size >= nbsym+1 else np.flipud(indmin[:])
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
            phi = k*np.pi/ndirs
            (indmin,indmax) = extr(np.real(np.exp(1j*phi)*r))
            ner[k] = len(indmin) + len(indmax) #NOTE: See how these two functions interact
        stop = any(ner < 3)
    else:
        (indmin, indmax) = extr(r,nargout=2)
        ner = len(indmin) + len(indmax)
        stop = ner < 3

    return stop

# computes the mean of the envelopes and the mode amplitude estimate
def mean_and_amplitude(m : np.ndarray, t : np.ndarray, INTERP : str, MODE_COMPLEX : int, ndirs : int, nargout : int = 4):      # returns (envmoy, nem, nzm, amp), 
    #TODO: fix the nargout to adapt to the multiple usages in this file
    NBSYM = 2
    nem = np.zeros(ndirs)
    nzm = np.zeros(ndirs)
    envmin = np.zeros(ndirs,dtype=object) if MODE_COMPLEX else np.zeros(ndirs)
    envmax = np.zeros(ndirs,dtype=object) if MODE_COMPLEX else np.zeros(ndirs)

    if MODE_COMPLEX:
        if MODE_COMPLEX == 1:
            for k in range(ndirs):
                phi = k*np.pi/ndirs
                y = np.real(np.exp(-1j*phi)*m)
                indmin, indmax, indzer = extr(y, nargout=3) 
                nem[k] = len(indmin) + len(indmax)
                nzm[k] = len(indzer)
                (tmin, tmax, zmin, zmax) = boundary_conditions(indmin, indmax, t, y, m, NBSYM)
                f = interp.interp1d(tmin,zmin,kind=INTERP)  
                envmin[k] = f(t)
                f = interp.interp1d(tmax,zmax,kind=INTERP)
                envmax[k] = f(t)
            envmoy = np.mean((envmin+envmax)/2,axis=0)
            if nargout > 3:
                amp = np.mean(np.abs(envmax-envmin),axis=0)/2
        elif MODE_COMPLEX == 2:
            for k in range(ndirs):
                phi = k*np.pi/ndirs
                y = np.real(np.exp(-1j*phi)*m)
                indmin, indmax, indzer = extr(y,nargout=3)
                nem[k] = len(indmin) + len(indmax)
                nzm[k] = len(indzer)
                (tmin, tmax, zmin, zmax) = boundary_conditions(indmin, indmax, t,y,y,NBSYM) 
                if 'quadratic' or 'cubic' or 'spline' in INTERP:
                    f = interp.interp1d(tmin,zmin,kind=INTERP)
                    envmin[k] = np.exp(1j*phi)*f(t)
                    f = interp.interp1d(tmax,zmax,kind=INTERP)
                    envmax[k] = np.exp(1j*phi)*f(t)
                elif 'linear' in INTERP:
                    envmin[k] = np.interp(t,tmin,zmin)
                    envmax[k] = np.interp(t,tmax,zmax)
            envmoy = np.mean((envmin+envmax),axis=0)
            if nargout > 3:
                amp = np.mean(np.abs(envmax-envmin),axis=0)/2
    else:
        (indmin, indmax, indzer) = extr(m, nargout=3)
        nem = len(indmin) + len(indmax)
        nzm = len(indzer)
        (tmin, tmax, mmin, mmax) = boundary_conditions(indmin, indmax, t, m, m, NBSYM)
        if 'linear' in INTERP:  
            envmin = np.interp(t, tmin, mmin)
            envmax = np.interp(t,tmax,mmax)
        elif 'quadratic' or 'cubic' or 'spline' in INTERP:
            f = interp.interp1d(tmin,mmin,kind=INTERP) # using quadratic spline interpolation
            envmin = f(t)
            f = interp.interp1d(tmax,mmax,kind=INTERP)
            envmax = f(t)
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
        if type(nem) == int:
            stop = not ((np.mean(sx > sd) > tol or (any(sx > sd2))) and ((nem > 2)))
        else:
            stop = not ((np.mean(sx > sd) > tol or (any(sx > sd2))) and all(nem > 2))   
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
    except Exception as e:
        print(e)
        moyenne = np.zeros((1,len(m)))
        stop = 1
    finally:
        return stop,moyenne

def stop_sifting_fixe_h(t:np.ndarray,m:np.ndarray,INTERP:str,stop_count:int,FIXE_H:int,MODE_COMPLEX:int,ndirs:int,nargout:int=2):
    try:
        (moyenne,nem,nzm) = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs,nargout=3)
        if isinstance(nem,float):
            nem = np.array(nem)
        if isinstance(nzm,float):
            nzm = np.array(nzm)
        # Zip arrays together, get each pairs difference, take the absolute value, and check if all greater than one.
        if (np.abs(np.diff(np.array(list(zip(nem,nzm))))) > 1).all():   
            stop = 0
            stop_count = 0
        else:
            stop_count = stop_count+1
            stop = (stop_count == FIXE_H)
    except:
        moyenne = np.zeros((1,len(m))) 
        stop = 1
        stop_count = 0 
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
def init(x,t,stop,ndirs,display_sifting,MODE_COMPLEX,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask):
    mask_signal=0
    
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    #default for t.
    if isinstance(t,int):
        t = np.arange(0,len(x))

    if x.ndim != 1:
        raise TypeError('X must have only one row or one column.')

    if t.ndim != 1:
        raise TypeError('Option field T must have only one row or one column')

    if not np.isreal(t.all()):
        raise TypeError('Time instants T must be a real vector')

    if t.shape[0] != x.shape[0]:
        raise TypeError('X and option field T must have the same length')

    if stop.ndim != 1 or stop.shape[0] > 3:
        raise TypeError('Option field STOP must have only one row or one column of max three elements')

    if not all(np.isfinite(x)):
        raise ValueError('Data elements must be finite')

    L = len(stop)
    if L < 3:
        stop[2] = 0.05
    if L < 2:
        stop[1] = 0.5

    if not isinstance(INTERP,str) or INTERP not in {'linear', 'cubic', 'quadratic','spline'}:
        raise TypeError('INTERP field must be linear, cubic, or spline')

    #special procedure when a masking signal is specified
    if isinstance(mask,np.ndarray):
        if mask.ndim != 1 or mask.shape[0] != x.shape[0]:
            raise TypeError('Masking signal must have the same dimension as the analyzed signal X')
        mask_signal = mask
        mask = 0
        imf1 = emd(x+mask_signal,
                    t=t,stop=stop,
                    ndirs=ndirs,
                    display_sifting=display_sifting,
                    MODE_COMPLEX=MODE_COMPLEX,
                    MAXITERATIONS=MAXITERATIONS,
                    FIXE=FIXE,
                    FIXE_H=FIXE_H,
                    MAXMODES=MAXMODES,
                    INTERP=INTERP,
                    mask=mask)
        imf2 = emd(x-mask_signal,
                    t=t,stop=stop,
                    ndirs=ndirs,
                    display_sifting=display_sifting,
                    MODE_COMPLEX=MODE_COMPLEX,
                    MAXITERATIONS=MAXITERATIONS,
                    FIXE=FIXE,
                    FIXE_H=FIXE_H,
                    MAXMODES=MAXMODES,
                    INTERP=INTERP,
                    mask=mask)
        if imf1.shape[0] != imf2.shape[0]:
            warnings.warn('emd:warning, the two sets of IMFs have different sizes: {} and {} IMFs.'.format(imf1.shape[0],imf2.shape[0]))
        S1 = imf1.shape[0]
        S2 = imf2.shape[0]
        if S1 != S2:
            if S1 < S2:
                tmp = imf1
                imf1 = imf2
                imf2 = tmp
            # need some code here to append a certain number of rows to imf2.
            row_gap = imf1.shape[0] - imf2.shape[0] # number of rows to add.
            for i in range(row_gap):
                imf2 = np.vstack((imf2,np.zeros(imf2.shape[1])))
        imf = (imf1+imf2)/2

    sd = stop[0]
    sd2 = stop[1]
    tol = stop[2]
    lx = x.shape[0]
    sdt = sd*np.ones(lx)
    sd2t = sd2*np.ones(lx)

    if FIXE:
        MAXITERATIONS = FIXE
        if FIXE_H:
            raise TypeError('cannot use both ''FIX'' and ''FIX_H'' modes')

    # if not real values found, use complex_version, false otherwise.
    MODE_COMPLEX = MODE_COMPLEX if not all (np.isreal(x)) else 0
    if MODE_COMPLEX != 0 and MODE_COMPLEX != 1 and MODE_COMPLEX != 2: 
        raise ValueError('COMPLEX_VERSION parameter must be equal to 0, 1, or 2')

    # number of extrema and zero-crossings in residual
    ner = lx
    nzr = lx

    r = x
    k=1
    nbit=0
    NbIt=0

    if not isinstance(mask_signal,np.ndarray): # if a masking signal is specified "imf" already exists at this stage
        imf = np.array([])
    else:
        mask = mask_signal

    return (x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask)