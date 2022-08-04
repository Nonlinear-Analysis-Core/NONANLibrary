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

1)  AMI_Stergiou, This is a histogram-based method of Average Mutual Information that is used to find a time lag using average mutual information for state space reconstruction.
2)  AMI_Thomas, This is a kernal density-based method of Average Mutual Information that is used to find a time lag for state space reconstruction.
3)  ChaosLibrary, This uses a number of systems of differential equations that can be used to create chaotic attractors.
4)  dfa, This is used to calculate an alpha by way of Detrended Fluctuation Analysis.
5)  Ent_Ap, Used to calculate the Approximate Entropy of a time series.
6)  Ent_MS_Plus, Used to calculate the Refined Composite Multiscale Sample Entropy, Composite Multiscale Entropy, Multiscale Entropy, Multiscale Fuzzy Entropy, and Generalized Multiscale Entropy of a time series.
7)  Ent_Permu, Used to calculate the Permutation Entropy of a time series using a log of base 2.
8)  Ent_Samp, Used to calculate the Sample Entropy of a time series.
9)  Ent_Symbolic, Used to calculate the Symbolic Entropy of a time series.
10) Ent_xAp, Uses Apprimate Entropy to calculate the Cross Approximate Entropy between two time series.
11) Ent_xSamp, Uses Sample Entropy to calculate the Cross Sample Entropy between two time series.
12) FNN, Calculates an embedding dimension for state space reconstruction using the method of False Nearest Neighbors.
13) LyE_R, Calculates the Largest Lyapunov Exponent using the method published by Rosenstein.
14) LyE_W, Calculates the Largest Lyapunov Exponent using the method published by Wolf.
15) RelPhase_Cont, This Continuous Relative Phase script can be used to find the phase between two cyclic time series.
16) RelPhase_Disc, This Discrete Relative Phase script can be used to find the phase between two discrete time series.
17) RQA, Can be used to perform Recurrance Quantification Analysis.
18) Surr_findrho, This should be used to find thee optimal noise level used in creating a Pseudo Period surrogate time series.
19) Surr_PseudoPeriodic, This is used to find a Pseudo Period surrogate time series using the result from Surr_findrho.
20) Surr_Theiler, This can be used to create different surrogates by the methods published by Theiler.

COPYRIGHT

Copyright 2021 Nonlinear Analysis Core, Center for Human Movement Variability, University of Nebraska at Omaha

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright  notice, this list of conditions and the following disclaimer in the  documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTACT

Please contact bmchnonan@unomaha.edu regarding any questions or troubleshooting.
