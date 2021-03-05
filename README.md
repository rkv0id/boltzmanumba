# boltzmanumba
<img src="./gpu_out/vel.0019.png">

A naive parallelization of sequential Lattice Boltzmann code picked up from a random GitHub gist.

LBM is a simulation technique for complex fluid systems. It is known to be so performance greedy while also being an embarassingly parallel algorithm. This repo is a basic attempt of distributing the LBM algorithm on a CUDA-able GPU using the [Numba API](http://numba.pydata.org).

The notebook provided within this repository contains a benchmarking of the GPU/CPU acceleration.