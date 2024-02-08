import numpy as np

def fgn_sim(n=1000, H=0.7):
    """Create Fractional Gaussian Noise
     Inputs:
            n: Number of data points of the time series. Default is 1000 data points.
            H: Hurst parameter of the time series. Default is 0.7.
     Outputs:
            An array of n data points with variability H
    # =============================================================================
                                ------ EXAMPLE ------
            
          - Create time series of 1000 datapoints to have an H of 0.7
          n = 1000
          H = 0.7
          dat = fgn_sim(n, H)
            
          - If you would like to plot the timeseries:
          import matplotlib.pyplot as plt
          plt.plot(dat)
          plt.title(f"Fractional Gaussian Noise (H = {H})")
          plt.xlabel("Time")
          plt.ylabel("Value")
          plt.show()
    # =============================================================================
    """    

    # Settings:
    mean = 0
    std = 1

    # Generate Sequence:
    z = np.random.normal(size=2*n)
    zr = z[:n]
    zi = z[n:]
    zic = -zi
    zi[0] = 0
    zr[0] = zr[0] * np.sqrt(2)
    zi[n-1] = 0
    zr[n-1] = zr[n-1] * np.sqrt(2)
    zr = np.concatenate([zr[:n], zr[n-2::-1]])
    zi = np.concatenate([zi[:n], zic[n-2::-1]])
    z = zr + 1j * zi

    k = np.arange(n)
    gammak = (np.abs(k - 1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k + 1)**(2*H)) / 2
    ind = np.concatenate([np.arange(n - 1), [n - 1], np.arange(n - 2, 0, -1)])
    gammak = gammak[ind]  # Circular shift of gammak to match n
    gkFGN0 = np.fft.ifft(gammak)
    gksqrt = np.real(gkFGN0)

    if np.all(gksqrt > 0):
        gksqrt = np.sqrt(gksqrt)
        z = z[:len(gksqrt)] * gksqrt
        z = np.fft.ifft(z)
        z = 0.5 * (n - 1)**(-0.5) * z
        z = np.real(z[:n])
    else:
        gksqrt = np.zeros_like(gksqrt)
        raise ValueError("Re(gk)-vector not positive")

    # Standardize: (z - np.mean(z)) / np.sqrt(np.var(z))
    ans = std * z + mean
    return ans
