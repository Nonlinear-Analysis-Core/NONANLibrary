import numpy as np
import matplotlib.pyplot as plt

def dfa(data, scales, order=1, plot=True):
    
    """Perform Detrended Fluctuation Analysis on data

    Inputs:
        data: 1D numpy array of time series to be analyzed.
        scales: List or array of scales to calculate fluctuations
        order: Integer of polynomial fit (default=1 for linear)
        plot: Return loglog plot (default=True to return plot)

    Outputs:
        scales: The scales that were entered as input
        fluctuations: Variability measured at each scale with RMS
        alpha value: Value quantifying the relationship between the scales 
                     and fluctuations
        
....References:
........Damouras, S., Chang, M. D., Sejdi, E., & Chau, T. (2010). An empirical 
..........examination of detrended fluctuation analysis for gait data. Gait & 
..........posture, 31(3), 336-340.
........Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
..........temporal correlations under extreme random dilution. Physica A: 
..........Statistical Mechanics and its Applications, 389(24), 5573-5580.
........Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
..........Quantification of scaling exponents and crossover phenomena in 
..........nonstationary heartbeat time series. Chaos: An Interdisciplinary 
..........Journal of Nonlinear Science, 5(1), 82-87.
# =============================================================================
                            ------ EXAMPLE ------

      - Generate random data
      data = np.random.randn(5000) 
      
      - Create a vector of the scales you want to use
      scales = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
      
      - Set a detrending order. Use 1 for a linear detrend.
      order = 1
      
      - run dfa function
      s, f, a = dfa(data, scales, order, plot=True)
# =============================================================================
"""
    
    # Check if data is a column vector (2D array with one column)
    if data.shape[0] == 1:
        # Reshape the data to be a column vector
        data = data.reshape(-1, 1)
    else:
        # Data is already a column vector
        data = data
    
# =============================================================================
##########################   START DFA CALCULATION   ##########################
# =============================================================================
    
    # Step 1: Integrate the data
    integrated_data = np.cumsum(data - np.mean(data))

    fluctuation = []

    for scale in scales:
        # Step 2: Divide data into non-overlapping window of size 'scale'
        chunks = len(data) // scale
        ms = 0.0

        for i in range(chunks):
            this_chunk = integrated_data[i*scale:(i+1)*scale]
            x = np.arange(len(this_chunk))
            
            # Step 3: Fit polynomial (default is linear, i.e., order=1)
            coeffs = np.polyfit(x, this_chunk, order)
            fit = np.polyval(coeffs, x)
            
            # Detrend and calculate RMS for the current window
            ms += np.mean((this_chunk - fit) ** 2)            

        # Calculate average RMS for this scale
        fluctuation.append(np.sqrt(ms / chunks))
        
        # Perform linear regression
    alpha, intercept = np.polyfit(np.log(scales), np.log(fluctuation), 1)

        
    # Create a log-log plot to visualize the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(scales, fluctuation, marker='o', markerfacecolor = 'red', markersize=8, 
                   linestyle='-', color = 'black', linewidth=1.7, label=f'Alpha = {alpha:.3f}')
        plt.xlabel('Scale (log)')
        plt.ylabel('Fluctuation (log)')
        plt.legend()
        plt.title('Detrended Fluctuation Analysis')
        plt.grid(True)
        plt.show()

    # Return the scales used, fluctuation functions and the alpha value
    return scales, fluctuation, alpha



