//https://github.com/rsnemmen/OpenCL-examples/blob/master/Hello_World/hello.c
//https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/
//https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h
//http://neuralnetworksanddeeplearning.com/chap1.html
//http://neuralnetworksanddeeplearning.com/chap2.html

#define NO_GPU
/*
  NO_GPU still uses OpenCL so that I don't have to make huge changes, it just does the work on the CPU instead of the GPU.
  It's not for computers that don't have a GPU, it's for debugging, as valgrind only detects bad operations on the CPU.
*/

#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "opencl_util.h"

//https://github.com/projectgalateia/mnist
#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "mnist.h"

#ifdef NO_GPU
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "compute.cl"
#endif

float scaled_rand() {
  return 2*(rand()/(float) RAND_MAX - 0.5);
}

unsigned scaled_urand(unsigned max) {
  return rand()/(RAND_MAX/max);
}

char* read_all(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) return NULL;
  
  fseek(fp, 0, SEEK_END);
  size_t length = ftell(fp);
  
  char *buf = malloc(length + 1);
  if (buf) {
    rewind(fp);
    fread(buf, 1, length, fp);
    buf[length] = '\0';
  }
  
  fclose(fp);
  return buf;
}

mnist_data** shuffle_data(mnist_data *data, unsigned data_length) {
  mnist_data **res = malloc(sizeof(mnist_data *)*data_length);
  if (!res) return NULL;
  
  for (mnist_data **cur = res, *cdata = data; cur < res + data_length; cur++, cdata++) {
    *cur = cdata;
  }
  for (mnist_data **cur = res; cur < res + data_length; cur++) {
    mnist_data **rand = res + scaled_urand(data_length);
    mnist_data *tmp = *rand;
    *rand = *cur;
    *cur = tmp;
  }
  
  return res;
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
    
    TODO: error_term is only ever needs to keep memory on the GPU, BUT currently keeps an unused buffer on the CPU side!
    ...HOWEVER, when NO_GPU is defined the memory on the CPU IS REQUIRED.
  */
  memory error_term;
  
  /*
    Store the sum of this layer's error over several training inputs
    
    TODO: as with error_term, this is only used in the GPU unless NO_GPU is defined.
  */
  memory bias_error; // adding this now (did a ctrl+h to replace it's previous name accumulated_error
  memory weight_error; // adding this now
  
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
  err = mem_init(&l->bias_error, size, sizeof(float), 0, context);
  if (err) goto err_bias_error_init;
  err = mem_init(&l->weight_error, size, sizeof(float), 0, context);
  if (err) goto err_weight_error_init;
  l->inputs = inputs;
  l->size = size;
  
  for (float *cur = l->weights.ptr; cur < ((float *) l->weights.ptr) + l->weights.length; cur++) {
    *cur = scaled_rand();
  }
  for (float *cur = l->biases.ptr; cur < ((float *) l->biases.ptr) + l->biases.length; cur++) {
    *cur = scaled_rand();
  }
#ifdef NO_GPU
  for (float *cur = l->error_term.ptr; cur < ((float *) l->error_term.ptr) + l->error_term.length; cur++) {
    *cur = 0;
  }
#endif
  for (float *cur = l->bias_error.ptr; cur < ((float *) l->bias_error.ptr) + l->bias_error.length; cur++) {
    *cur = 0;
  }
  for (float *cur = l->weight_error.ptr; cur < ((float *) l->weight_error.ptr) + l->weight_error.length; cur++) {
    *cur = 0;
  }
  
  return 0;
  
err_weight_error_init:
  mem_destroy(&l->bias_error);
err_bias_error_init:
  mem_destroy(&l->error_term);
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
  mem_destroy(&l->error_term);
  mem_destroy(&l->bias_error);
  mem_destroy(&l->weight_error);
}

typedef struct {
  layer *layers;
  unsigned size;
  float learning_rate;
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

int layer_compute(layer *l, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
#ifdef NO_GPU
  fn_global_id = 0;
  for (unsigned i = 0; i < l->size; i++) {
    compute_output_and_err_part(l->inputs->ptr, l->inputs->length, l->outputs.ptr, l->weights.ptr, l->biases.ptr, l->error_term.ptr, l->size);
  }
  return 0;
#else
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
#endif
}

int network_compute(network *n, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  int err;
  layer *cur = n->layers;
  
  for (; cur < n->layers + n->size; cur++) {
    err = layer_compute(cur, max_workgroup_size, commands, kernel);
    if (err) return err;
#ifndef NO_GPU
    clFinish(commands);
#endif
    //printf("\n\n"); //debug
  }

#ifdef NO_GPU
  return 0;
#else
  cur--;
  return mem_read_buffer(&cur->outputs, commands, CL_TRUE);
#endif
}

int layer_finish_compute_err(layer *l, layer *next, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
#ifdef NO_GPU
  fn_global_id = 0;
  for (unsigned i = 0; i < l->size; i++) {
    backpropagate_err((float *) l->error_term.ptr, (float *) l->bias_error.ptr, l->size, (float *) next->weights.ptr, (float *) next->error_term.ptr, next->size);
  }
  return 0;
#else
  int err;
  
  size_t total_workgroup_size = l->size;
  total_workgroup_size += max_workgroup_size - total_workgroup_size%max_workgroup_size;
  
  //printf("\n\n%u -> %ld\n\n", l->size, total_workgroup_size);
  
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &l->error_term.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &l->bias_error.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 2, sizeof(unsigned), &l->size);
  if (err) return err;
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &next->weights.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &next->error_term.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 5, sizeof(unsigned), &next->size);
  if (err) return err;
  
  
  return clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &total_workgroup_size, &max_workgroup_size, 0, NULL, NULL);
#endif
}

int network_compute_err(network *n, float *answer, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  int err;
  layer *cur = n->layers + n->size - 1;
  
  /*
    Since the final layer of the network is a special case, it is run on the CPU so I have to do less programming.
    This is (probably) suboptimal and should be addressed by future me.
    I also should figure out the overhead of running a kernel.
  */
#ifndef NO_GPU
  err = mem_read_buffer(&cur->error_term, commands, CL_TRUE);
  if (err) return err;
#endif
  
  for (unsigned i = 0; i < cur->size; i++) {
    float *term_addr = i + (float *) cur->error_term.ptr;
    *term_addr = (*term_addr) * (*(i + (float *) cur->outputs.ptr) - answer[i]);
    //printf("%f ", *term_addr);
  }
  //printf("\n\n");
#ifndef NO_GPU
  err = mem_write_buffer(&cur->error_term, commands, CL_TRUE);
  if (err) return err;
#endif
  cur--;
  
  for (; cur >= n->layers; cur--) {
    err = layer_finish_compute_err(cur, cur + 1, max_workgroup_size, commands, kernel);
    if (err) return err;
#ifndef NO_GPU
    clFinish(commands);
#endif
  }
  
  return 0;
}

int layer_update_neurons(layer *l, float learning_rate, unsigned mini_batch_size, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
#ifdef NO_GPU
  fn_global_id = 0;
  for (unsigned i = 0; i < l->size; i++) {
    unsigned prev_size = l->inputs->length;
    update_neurons(l->weights.ptr, l->biases.ptr, l->bias_error.ptr,  l->size, l->inputs->ptr, prev_size, learning_rate, mini_batch_size);
  }
  return 0;
#else
  int err;
  
  size_t total_workgroup_size = l->size;
  total_workgroup_size += max_workgroup_size - total_workgroup_size%max_workgroup_size;
  
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &l->weights.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &l->biases.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &l->bias_error.buf);
  if (err) return err;
  err = clSetKernelArg(kernel, 3, sizeof(unsigned), &l->size);
  if (err) return err;
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &l->inputs->buf);
  if (err) return err;
  unsigned prev_size = l->inputs->length;
  err = clSetKernelArg(kernel, 5, sizeof(unsigned), &prev_size);
  if (err) return err;
  err = clSetKernelArg(kernel, 6, sizeof(float), &learning_rate);
  if (err) return err;
  err = clSetKernelArg(kernel, 7, sizeof(unsigned), &mini_batch_size);
  if (err) return err;
  
  
  return clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &total_workgroup_size, &max_workgroup_size, 0, NULL, NULL);
#endif
}

int network_update_neurons(network *n, float learning_rate, unsigned mini_batch_size, size_t max_workgroup_size, cl_command_queue commands, cl_kernel kernel) {
  int err;
  
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    err = layer_update_neurons(cur, learning_rate, mini_batch_size, max_workgroup_size, commands, kernel);
    if (err) return err;
#ifndef NO_GPU
    clFinish(commands);
#endif
  }
  return 0;
}

int network_prep_neuron_params(network *n, cl_command_queue commands) { // ok the whole function name should change
  int err = 0;
  err = mem_write_buffer(n->layers[0].inputs, commands, CL_FALSE); // should this guy really be here or left up to the caller?
  if (err) return err;
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    err = mem_write_buffer(&cur->weights, commands, CL_FALSE);
    if (err) break;
    err = mem_write_buffer(&cur->biases, commands, CL_FALSE);
    if (err) break;
    err = mem_write_buffer(&cur->bias_error, commands, CL_FALSE);
    if (err) break;
    err = mem_write_buffer(&cur->weight_error, commands, CL_FALSE);
    if (err) break;
  }
  clFinish(commands);
  return err;
}

int network_train(network *n, mnist_data *data, unsigned data_length, float learning_rate, unsigned mini_batch_size, size_t max_workgroup_size, cl_command_queue commands, cl_kernel layer_compute_kernel, cl_kernel layer_err_compute_kernel, cl_kernel layer_update_kernel) {
  int err = 0;
  float answer[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  mnist_data **shuffled = shuffle_data(data, data_length);
  mnist_data **cur = shuffled;
  
  for (mnist_data **end = shuffled + mini_batch_size; end <= shuffled + data_length; end += mini_batch_size) {
    unsigned num_correct = 0;
    for (; cur < end; cur++) {
      int last_answer_index = (*cur)->label;
      answer[last_answer_index] = 1;
      n->layers[0].inputs->ptr = &(*cur)->data[0][0];
      
      for (unsigned i = 0; i < n->layers[0].inputs->length; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", *(((float *) n->layers[0].inputs->ptr) + i) >= 0.5 ? '1' : '0');
      }
      printf("\n");
      
#ifndef NO_GPU
      err = network_prep_neuron_params(n, commands);
      if (err) return err;
#endif
      err = network_compute(n, max_workgroup_size, commands, layer_compute_kernel);
      if (err) return err;
      
      printf("label: %d output: [", last_answer_index);
      layer *last = &n->layers[n->size-1];
      int guess = -1;
      float max = 0;
      for (unsigned i = 0; i < last->size; i++) {
        float out = *(((float *) last->outputs.ptr) + i);
        if (out > max) {
          guess = i;
          max = out;
        }
        printf("%f ", out);
      }
      printf("\b] guess: %d\n", guess);
      if (guess == last_answer_index) num_correct++;
      
      err = network_compute_err(n, &answer[0], max_workgroup_size, commands, layer_err_compute_kernel);
      if (err) return err;
      answer[last_answer_index] = 0;
    }
    printf("%u/%u\n", num_correct, mini_batch_size);
    err = network_update_neurons(n, learning_rate, mini_batch_size, max_workgroup_size, commands, layer_update_kernel);
    if (err) return err;
    //return 0; //TODO: finish function and remove this debug line
  }
  
  return err;
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
  
  char *kernel_source = read_all("./compute.cl");
  if (!kernel_source) PANIC("read_all(\"./compute.cl\")", -28);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
  free(kernel_source);
  if (err) PANIC("clCreateProgramWithSource", err);
  
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err) {
    size_t len = 0;
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    if (err) PANIC("clBuildProgram, clGetProgramBuildInfo", err);
    char *buffer = malloc(len);
    if (!buffer) PANIC("clBuildProgram, malloc", -28);
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    if (err) PANIC("clBuildProgram, clGetProgramBuildInfo", err);
    printf("%s\n", buffer);
    free(buffer);
    PANIC("clBuildProgram", err);
  }
  
  cl_kernel layer_compute_kernel = clCreateKernel(program, "compute_output_and_err_part", &err);
  if (err) PANIC("clCreateKernel", err);
  cl_kernel layer_err_compute_kernel = clCreateKernel(program, "backpropagate_err", &err);
  if (err) PANIC("clCreateKernel", err);
  cl_kernel layer_update_kernel = clCreateKernel(program, "update_neurons", &err);
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
  
  /*
  for (unsigned i = 0; i < input.length; i++) {
    if (i % 28 == 0) printf("\n");
    printf("%c", *(((float *) input.ptr) + i) >= 0.5 ? '1' : '0');
  }
  printf("\nlabel: %d\n", data[0].label);
  */
  
  network n;
  unsigned layer_sizes[] = {15, 10, 0};
  err = network_init(&n, &input, &layer_sizes[0], context);
  // move the following to network_init?
  n.learning_rate = 3.0;
  
  size_t max_workgroup_size;
  // TODO: is the max_workgroup_size specific to a kernel? Would it differ between layer_compute_kernel and layer_err_compute_kernel?
  err = clGetKernelWorkGroupInfo(layer_compute_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
  if (err) PANIC("clGetKernelWorkGroupInfo", err);
  
  float learning_rate = 3.0;
  unsigned mini_batch_size = 10;
  err = network_train(&n, data, data_count, learning_rate, mini_batch_size, max_workgroup_size, commands, layer_compute_kernel, layer_err_compute_kernel, layer_update_kernel);
  printf("%d\n", err);
  
  free(data);
  network_destroy(&n);
  clReleaseKernel(layer_compute_kernel);
  clReleaseKernel(layer_err_compute_kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  return 0;
}

#ifdef NO_GPU
#pragma GCC diagnostic pop
#endif
