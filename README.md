# NN

## To build

Clone this repository then `make`.

## To use

`./network`  
After each epoch, the neural network will be tested and the program will report how accurate it is.

## Some notes

* At first, this project used OpenGL to run code on the GPU, but I removed that capability early on to make it easier to debug. Also, since the network I'm using this for is so small, the overhead of using the GPU made it significantly slower than not using the GPU. This feature will be re-implemented at a later date.

* The MNIST dataset is loaded using code by Nuri Park which can be found at https://github.com/projectgalateia/mnist

* I'm convinced there's a bug somewhere that's inhibiting proper training and I will find it and fix it

* A lot of important parameters like learning rate, mini batch size, the number of epochs etc. are hard coded. It's not a priority right now for me to fix this since I'm working on debugging

* The neural network can only use sigmoid neurons and the quadratic cost function. I'll get to allowing more options once bugs are fixed
