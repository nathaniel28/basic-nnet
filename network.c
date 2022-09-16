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
#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "mnist.h"

const char *kernel_source = 
"kernel void compute(global float *input, const unsigned input_count, global float *output, global float *weights, global float *bias, global float *error_term, const unsigned layer_size) {\n"
"  int id = get_global_id(0);\n"
"  if (id >= (int) layer_size) return;\n"
"  output += id;\n"
"  weights += id*input_count;\n"
"  bias += id;\n"
"  error_term += id;\n"
"  float res = 0;\n"
"  global const float *end = input + input_count;\n"
"  while (input < end) {\n"
"    res += (*input++) * (*weights++);\n"
"  }\n"
"  res += *bias;\n"
"  float raised = exp(-res);\n"
"  res = 1/(1+raised);\n"
"  *output = res;\n"
"  *error_term = raised*res*res;\n"
"  printf(\"(res:%f output:%f) \", *error_term, *output);\n" //debug
"}\n"
;

float scaled_rand() {
  return 2*(rand()/(float) RAND_MAX - 0.5);
}

typedef struct {
  memory *inputs; // possibly a pointer to the output of another layer; this layer will not modify nor free it
  
  /*
    the bias of the ith neuron is at (((float *) biases.ptr) + i)
    the jth weight of the ith neuron is at (((float *) weights.ptr) + i*inputs->length + j)
  */
  memory weights, biases;
  
  /*
    the output of this layer called l, computed using σ(wˡaˡ⁻¹ + bˡ) where
    σ(v) = 1/(1+exp(-v))
    wˡ = weights of layer l
    aˡ⁻¹ = output of layer l-1
    bˡ = biases of layer l
  */
  memory outputs;
  
  /*
    THIS BUFFER IS USED MORE THAN ONCE, AND STORES AN INTERMIDIATE TERM BEFORE THE FINAL ANSWER.
    furthermore, it is only used when a neural network is training, so use a different kernel if not
    first, it stores sigmoid prime of weighted input in other words, σ′(zˡ) where
    zˡ = wˡaˡ⁻¹ + bˡ
    σ′(v) = exp(-v)/((1+exp(-v))^2)
    this first value is computed in the same kernel as outputs for efficiency
    second and finally, it stores error
    if this is the output layer, it becomes ∇ₐC⊙σ′(zˡ) where
    . ∇ₐC = (aᴸ - y) ASSUMING THE COST FUNCTION IS QUADRATIC COST where
    aᴸ = the output of the final layer in the network
    y = the expected output
    if this is not the output layer, it becomes ((wˡ⁺¹)ᵀδˡ⁺¹)⊙σ′(zˡ) where
    wˡ⁺¹ = weights of layer l+1
    δˡ⁺¹ = error of layer l+1
    σ′(zˡ) = what is currently kept here but about to be overwritten
    in short: at first this is σ′(zˡ), then it becomes δˡ
  */
  memory error_term;
  // TODO: a lot of data here only ever needs to live on the GPU, BUT currently keeps an unused buffer on the CPU side!
  
  unsigned size; // number of neurons in this layer
} layer;

int layer_init(layer *l, unsigned size, memory *inputs, cl_context context) {
  int err;
  
  err = mem_init(&l->weights, inputs->length*size, sizeof(float), 0, context);
  if (err) return err;
  err = mem_init(&l->biases, inputs->length, sizeof(float), 0, context);
  if (err) goto err_biases_init;
  err = mem_init(&l->outputs, size, sizeof(float), 0, context);
  if (err) goto err_output_init;
  err = mem_init(&l->error_term, size, sizeof(float), 0, context);
  if (err) goto err_error_term_init;
  l->inputs = inputs;
  l->size = size;
  
  for (float *cur = l->weights.ptr; cur < ((float *) l->weights.ptr) + l->weights.length; cur++) {
    *cur = scaled_rand();
  }
  for (float *cur = l->biases.ptr; cur < ((float *) l->biases.ptr) + l->biases.length; cur++) {
    *cur = scaled_rand();
  }
  
  return 0;
  
err_error_term_init:
  mem_destroy(&l->outputs);
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

int network_prep_neuron_params(network *n, cl_command_queue commands) {
  int err = 0;
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    err = mem_write_buffer(&cur->weights, commands, CL_FALSE);
    if (err) break;
    err = mem_write_buffer(&cur->biases, commands, CL_FALSE);
    if (err) break;
  }
  clFinish(commands);
  return err;
}

int layer_compute(layer *l, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  int err;
  
  size_t total_workgroup_size = l->size;
  total_workgroup_size += max_workgroup_size - total_workgroup_size%max_workgroup_size;
  
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
  err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &l->error_term.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 6, sizeof(unsigned), &l->size);
  if (err) return err;
  
  return clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &total_workgroup_size, &max_workgroup_size, 0, NULL, NULL);
}

int network_compute(network *n, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  int err;
  layer *cur = n->layers;
  
  err = mem_write_buffer(cur->inputs, commands, CL_TRUE);
  if (err) return err;
  
  for (; cur < n->layers + n->size; cur++) {
    err = layer_compute(cur, max_workgroup_size, commands, kernel);
    if (err) return err;
    clFinish(commands);
    printf("\n\n"); //debug
  }
  
  cur--;
  return mem_read_buffer(&cur->outputs, commands, CL_TRUE);
}

float partial_quadratic_cost(float *expected, float *result, unsigned length) {
  float sum = 0;
  for (unsigned i = 0; i < length; i++) {
    float diff = (*expected++) - (*result++);
    sum += diff*diff;
  }
  return sum/2.0;
}

/*
int network_train(network *n, mnist_data *data, unsigned data_length, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  for (mnist_data *cur = data; cur < data + data_length; cur++) {
    
  }
}
*/

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
  
  mnist_data *data;
  unsigned data_count;
  err = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &data, &data_count);
  if (err) PANIC("mnist_load", err);
  memory input;
  input.ptr = &data[0].data[0];
  input.length = 28*28;
  input.unit_size = sizeof(float);
  input.buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, input.length*input.unit_size, input.ptr, &err);
  if (err) PANIC("clCreateBuffer", err);
  
  for (unsigned i = 0; i < input.length; i++) {
    if (i % 28 == 0) printf("\n");
    printf("%c", *(((float *) input.ptr) + i) >= 0.5 ? '1' : '0');
  }
  printf("\nlabel: %d\n", data[0].label);
  
  network n;
  unsigned layer_sizes[] = {15, 10, 0};
  err = network_init(&n, &input, &layer_sizes[0], context);
  
  size_t max_workgroup_size;
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
  if (err) PANIC("clGetKernelWorkGroupInfo", err);
  
  err = network_prep_neuron_params(&n, commands);
  if (err) PANIC("network_prep_neuron_params", err);
  err = network_compute(&n, max_workgroup_size, commands, kernel);
  if (err) PANIC("network_compute", err);
  
  free(data);
  network_destroy(&n);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  
  return 0;
}
