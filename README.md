# NONANToolbox

## INSTALLATION

To download this library, click on the green dropdown box that says "Code" and click on "Download ZIP". Once the ZIP file has downloaded, you will have to extract the files from that ZIP folder. Once the files are extracted, they will be available for your use.

### MATLAB VERSION

There are no known incompatibilities using MATLAB version R2019a.

MATLAB Toolboxes Required:

  Statistics and Machine Learning Toolbox
  
  Signal Processing Toolbox
  
  Image Processing Toolbox

  Parallel Computing Toolbox

  Application Compiler Toolbox 

### PYTHON VERSION  

To install from the requirements.txt file, make sure you have the package installer for Python (pip) on your PATH, and use the following command:

```
pip install -r requirements.txt
```

If you are installing using pip3, then simply use the command:

```
pip3 install -r requirements.txt
```

After installing these libraries, the Python scripts are available for use.

### DOCUMENTATION

For documentation related to this library, we have a GitHub page hosted [here](https://nonlinear-analysis-core.github.io/NONANLibrary/index.html).

### FILES

This is a list of the included functions and the full name of the methods.

All function and file names are lower_snake_case. The previous names still
work through shims in `matlab/deprecated/`; add that folder to your path if
you need them, and they will warn once per session.

| function | description |
|---|---|
| `ami` | Average mutual information versus lag, for choosing an embedding delay. Wrapper over `ami_histogram` and `ami_kde`. |
| `ami_histogram` | AMI by equal-width joint histogram. |
| `ami_kde` | AMI by Gaussian kernel density estimate. |
| `chaos_library` | Systems of differential equations that produce chaotic attractors. |
| `corr_dim` | Correlation dimension. |
| `crqa` | Cross recurrence quantification analysis. |
| `dfa` | Detrended fluctuation analysis. |
| `embed` | Delay embedding of a time series. |
| `ent_ap` | Approximate entropy. |
| `ent_ms_plus` | Refined composite multiscale, composite multiscale, multiscale, multiscale fuzzy, and generalized multiscale entropy. |
| `ent_permu` | Permutation entropy, log base 2. |
| `ent_samp` | Sample entropy. |
| `ent_symbolic` | Symbolic entropy. |
| `ent_weighted` | Weighted entropy of a recurrence plot. |
| `ent_xap` | Cross approximate entropy between two series. |
| `ent_xsamp` | Cross sample entropy between two series. |
| `fgn_sim` | Simulate fractional Gaussian noise at a specified Hurst exponent. |
| `fnn` | Embedding dimension by false nearest neighbours. |
| `jrqa` | Joint recurrence quantification analysis. |
| `line_hist` | Diagonal and vertical line histograms of a recurrence plot. |
| `lye_r` | Largest Lyapunov exponent, Rosenstein's method. Returns the divergence curve. |
| `lye_w` | Largest Lyapunov exponent, Wolf's method. Returns bits per unit time. |
| `mdrqa` | Multidimensional recurrence quantification analysis. |
| `psr` | Phase space reconstruction. |
| `rel_phase_cont` | Continuous relative phase between two cyclic series. |
| `rel_phase_disc` | Discrete relative phase between two series. |
| `rqa` | Recurrence quantification analysis. |
| `rqa_legacy` | Previous combined RQA/cRQA/jRQA/mdRQA entry point, kept for reproducibility. |
| `rqa_plot` | Plot a recurrence plot and its statistics. |
| `set_radius` | Find the radius giving a target percent recurrence. |
| `surr_find_rho` | Optimal noise radius for a pseudo-periodic surrogate. |
| `surr_pseudo_periodic` | Pseudo-periodic surrogate, using the radius from `surr_find_rho`. |
| `surr_theiler` | Theiler surrogates: shuffle, Fourier transform, and amplitude-adjusted Fourier transform. |

### TESTS

```
matlab -batch "addpath('tests/matlab'); run_tests"
python3 tests/python/run_tests.py
```

Headless, base MATLAB only, exits nonzero on failure. See `tests/README.md`.

COPYRIGHT

Copyright 2021 Nonlinear Analysis Core, Center for Human Movement Variability, University of Nebraska at Omaha

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright  notice, this list of conditions and the following disclaimer in the  documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTACT

Please contact bmchnonan@unomaha.edu regarding any questions or troubleshooting.
