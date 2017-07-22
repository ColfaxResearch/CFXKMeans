# CFXKMeans
CFXKMeans library is an open-source, optimized library for K-means clustering analysis developed by [Colfax Research](https://colfaxresearch.com).
Development of the library is described in detail in the white paper, "Optimization of Hamerly’s K-Means Clustering Algorithm: CFXKMeans Library" [webpage](https://colfaxresearch.com/cfxkmeans).

# Installation
#### C/C++ API

The installation requires either a GNU C/C++ Compiler or Intel C/C++ Compiler with OpenMP 4.0 support.
It has been verified on Intel Compiler 2017 and GCC-6.3.0.
To compile CFXKMeans, first configure the Makefile by selecting the compiler and the vecorization flags for your architecture:
```
########### Intel compiler ###################
CC=icc
CXX = icpc
### Enable vectorization support
## Intel Xeon Phi processor        -> -xMIC_AVX512
## Intel Xeon processor (HSW, BDW) -> -xCORE_AVX2
## Intel Xeon processor (SNB, IVB) -> -xAVX
VECFLAG=-xMIC_AVX512
CXXFLAGS=$(COMMONFLAGS) $(VECFLAG) -qopenmp
########### GCC (> 4.9.1) ####################
```
Then run make:
```
%  make
```

The archive file will be created in ```lib/``` directory, and the header file is in ```include/```.

#### Python API
Python API requires python and cython to compile. Numpy is rquired for CFXKMeans library.
The Makefile has support for compiling python extension:
```
%  make python-build     # creates a build/ with the extension
%  make python-install   # attempts to install
```

If you choose to use the setup.py manually, remember to set ```CC``` and ```CXX`` to the same compilers used for making the archive file.

# Usage
#### C/C++
See ```includes/kmeans.h``` for the definition of the ```KMeans``` class.
Example coming soon.

#### Python
Example usage can be found in  ```example/kmeans.py```.
```
import cfxkmeans
cfxk = cfxkmeans.KMeans(k, init).fit(data)
print(cfxk.labels_)
print(cfxk.centroids_)
```
Note that, currently, the library does not support creating the initial centroids. 
See the ```example/kmeans.py``` for an example on how to create inital centroids. 

# Running the application
CFXKMeans library uses OpenMP parallel framework for threading.
OpenMP framework supports various environment variables to control threading.
The performance of the library can be drastically different for some systems if the environment variables are not set correctly.

For Intel Xeon Phi processors we have found the folloing to give the best performance
```
% # For Xeon Phi 7250 processor
% export OMP_NUM_THREADS=68 # 1 thread per core
% export OMP_PLACES=cores
% export OMP_PROC_BIND=spread
% numactl -m 1 python kmeans.py #Using MCDRAM
```
For Intel Xeon processors we have found the folloing to give the best performance
```
% # For Xeon Phi 7250 processor
% export OMP_PLACES=cores
% export OMP_PROC_BIND=spread
% python kmeans.py
```

# Design and TODO
CFXKMeans library was originally developed as an excersize in code modernization (see [white paper](https://colfaxresearch.com/cfxkmeans)) of the computational workload (minimizing within-cluster variance).
So features such as choosing the initial centroids or computing the final variance total have not been implemented yet.
These features may be added in the future.
