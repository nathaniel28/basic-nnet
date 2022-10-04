# NN

## To build

Clone this repository then `make`.

## To use

`./network`  
After each epoch, the neural network will be tested and the program will report how accurate it is.

## Missing features

* Needs to be able to use better cost functions, only uses quadratic cost right now (I'm working on adding cross-entropy cost)

* Maybe should be able to use something besides sigmoid neurons but sigmoid neurons seem pretty good

* No command line args, currently hyperparameters and data/save file paths are hard coded (not a difficult change, but I'm working on improving the network's performance

## Some notes

* At first, this project used OpenGL to run code on the GPU, but I removed that capability early on to make it easier to debug. Also, since the network I'm using this for is so small, the overhead of using the GPU made it significantly slower than not using the GPU. This feature will be re-implemented at a later date.

* network\_compute, network\_compute\_err, and network\_update\_neurons all use pointer arithmetic because they were adapted from OpenCL kernels I had written

* The MNIST dataset is loaded using code by Nuri Park which can be found at https://github.com/projectgalateia/mnist
