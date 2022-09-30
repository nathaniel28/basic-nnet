# NN

## To build

Clone this repository then `make`.

## To use

`./network`  
After each epoch, the neural network will be tested and the program will report how accurate it is.

## Some notes

* At first, this project used OpenGL to run code on the GPU, but I removed that capability early on to make it easier to debug. Also, since the network I'm using this for is so small, the overhead of using the GPU made it significantly slower than not using the GPU.

* The MNIST dataset is loaded using code by Nuri Park which can be found at https://github.com/projectgalateia/mnist
