# boltzmanumba
<img src="./gpu_out/vel.0019.png">

Parallelization of a sequential Lattice Boltzmann code picked up from a random GitHub gist.

LBM is a simulation technique for complex fluid systems. It is known to be so performance greedy while also being an embarassingly parallel algorithm. This repo is a basic attempt of distributing the LBM algorithm on a CUDA-able GPU using the [Numba API](http://numba.pydata.org).

The `lattice_gpu_naive.py` file contains a naive parallelization of the code while the `lattice_gpu_opt.py` file contains a *memory-usage-optimized* code for the same algorithm.

The notebook provided within this repository contains a benchmarking of the GPU/CPU acceleration.