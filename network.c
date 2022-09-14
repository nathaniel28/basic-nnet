//https://github.com/rsnemmen/OpenCL-examples/blob/master/Hello_World/hello.c
//https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/
//https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h
//http://neuralnetworksanddeeplearning.com/chap1.html
//http://neuralnetworksanddeeplearning.com/chap2.html

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "opencl_util.h"

//https://github.com/projectgalateia/mnist
//#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "mnist.h"

const char *kernel_source = 
"kernel void compute(global float *input, const unsigned input_count, global float *output, global float *weights, global float *biases, const unsigned layer_size) {\n"
"  int id = get_global_id(0);\n"
"  if (id >= (int) layer_size) return;\n"
"  output += id;\n"
"  global float *weight = weights + id*input_count;\n"
"  float res = 0;\n"
"  global const float *end = input + input_count;\n"
"  while (input < end) {\n"
//"printf(\"[%f %f] \", *input, *weight);\n"
"    res += (*input++) * (*weight++);\n"
"  }\n"
"  res += *(biases + id);\n"
"  *output = 1/(1+exp(-res));\n"
"  printf(\"(res:%f output:%f) \", res, *output);\n"
"}\n"
;

float scaled_rand() {
  return 2*(rand()/(float) RAND_MAX - 0.5);
}

typedef struct {
  memory *inputs;
  /*
    the bias of the ith neuron is at (((float *) biases.ptr) + i)
    the jth weight of the ith neuron is at (((float *) weights.ptr) + i*inputs->length + j)
  */
  memory weights, biases;
  memory outputs;
  unsigned size; // number of neurons in this layer
} layer;

int layer_init(layer *l, unsigned size, memory *inputs, cl_context context) {
  int err;
  
  err = mem_init(&l->weights, inputs->length*size, sizeof(float), 0, context);
  if (err) return err;
  err = mem_init(&l->biases, inputs->length, sizeof(float), 0, context);
  if (err) goto err_biases_init;
  err = mem_init(&l->outputs, inputs->length, sizeof(float), 0, context);
  if (err) goto err_output_init;
  l->inputs = inputs;
  l->size = size;
  
  for (float *cur = l->weights.ptr; cur < ((float *) l->weights.ptr) + l->weights.length; cur++) {
    *cur = scaled_rand();
  }
  for (float *cur = l->biases.ptr; cur < ((float *) l->biases.ptr) + l->biases.length; cur++) {
    *cur = scaled_rand();
  }
  
  return 0;
  
err_output_init:
  mem_destroy(&l->biases);
err_biases_init:
  mem_destroy(&l->weights);
  return err;
}

void layer_destroy(layer *l) {
  mem_destroy(&l->outputs);
  mem_destroy(&l->biases);
  mem_destroy(&l->weights);
}

int prep_compute_kernel_args(layer *l, cl_kernel kernel) {
  int err = 0;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &l->inputs->buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 1, sizeof(unsigned), &l->inputs->length);
  if (err) return err;
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &l->outputs.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &l->weights.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &l->biases.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 5, sizeof(unsigned), &l->size);
  return err;
}

typedef struct {
  layer *layers;
  unsigned size;
} network;

int network_init(network *n, memory *input, unsigned *layer_sizes, cl_context context) {
  int err = 0;
  
  n->layers = malloc(sizeof(layer));
  if (!n->layers) return -1;
  
  err = layer_init(n->layers, *layer_sizes, input, context);
  if (err) goto err_primary_layer_init;
  n->size = 1;
  layer_sizes++;
  
  while (*layer_sizes) {
    layer *realloced = realloc(n->layers, sizeof(layer)*(n->size + 1));
    if (!realloced) {
      err = -2;
      goto err_layer_reallocation;
    }
    n->layers = realloced;
    err = layer_init(&n->layers[n->size], *layer_sizes, &n->layers[n->size - 1].outputs, context);
    if (err) {
      err = -3;
      goto err_layer_reallocation;
    }
    n->size++;
    layer_sizes++;
  }
  
  return 0;
  
err_layer_reallocation:
  n->size--;
  do {
    layer_destroy(&n->layers[n->size]);
  } while (n->size--);
err_primary_layer_init:
  free(n->layers);
  return err;
}

void network_destroy(network *n) {
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    layer_destroy(cur);
  }
  free(n->layers);
}

#define PANIC(msg, code) { printf("%s failed with error code %d.\n", msg, code); exit(-1); }

int main() {
  int err;
  cl_device_id device_id;
  
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (err) PANIC("clGetDeviceIDs", err);
  
  cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (err) PANIC("clCreateContext", err);
  
  cl_command_queue commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
  if (err) PANIC("clCreateCommandQueue", err);
  
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
  if (err) PANIC("clCreateProgramWithSource", err);
  
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err) {
    size_t len = 0;
    printf("%d ", clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
    char *buffer = malloc(len);
    printf("%d\n", clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL));
    printf("%s\n", buffer);
    free(buffer);
    PANIC("clBuildProgram", err);
  }
  
  cl_kernel kernel = clCreateKernel(program, "compute", &err);
  if (err) PANIC("clCreateKernel", err);
  
  /*
  mnist_data *data;
  unsigned data_count;
  err = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &data, &data_count);
  if (err) PANIC("mnist_load", err);
  */
  
  memory input;
  mem_init(&input, 28*28, sizeof(float), 0, context);
  for (float *cur = (float *) input.ptr; cur < ((float *) input.ptr) + input.length; cur++) {
    *cur = scaled_rand();
  }
  
  network n;
  unsigned layer_sizes[] = {15, 10, 0};
  err = network_init(&n, &input, &layer_sizes[0], context);
  
  /*
  size_t max_wg_size;
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
  if (err) PANIC("clGetKernelWorkGroupInfo", err);
  */
  
  err = mem_write_buffer(n.layers[0].inputs, commands, CL_FALSE);
  if (err) PANIC("mem_write_buffer", err);
  err = mem_write_buffer(&n.layers[0].weights, commands, CL_FALSE);
  if (err) PANIC("mem_write_buffer", err);
  err = mem_write_buffer(&n.layers[0].biases, commands, CL_FALSE);
  if (err) PANIC("mem_write_buffer", err);
  clFinish(commands);
  
  err = prep_compute_kernel_args(&n.layers[0], kernel);
  if (err) PANIC("prep_compute_kernel_args", err);
  
  size_t max_wg_size = 15;
  
  size_t total_size = n.layers[0].size;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &total_size, &max_wg_size, 0, NULL, NULL);
  if (err) PANIC("clEnqueueNDRangeKernel", err);
  
  clFinish(commands);
  
  //free(data);
  network_destroy(&n);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  
  return 0;
}
