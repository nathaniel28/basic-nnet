//http://neuralnetworksanddeeplearning.com/chap1.html
//http://neuralnetworksanddeeplearning.com/chap2.html

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "util.h"

//https://github.com/projectgalateia/mnist
#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "mnist.h"

float fdot(float *a, float *b, size_t len) {
  float res = 0;
  float *end = a + len;
  while (a < end) {
    res += (*a++) * (*b++);
  }
  return res;
}

float scaled_rand() {
  return 2*(rand()/(float) RAND_MAX - 0.5);
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
  
  for (mnist_data **cur = res; cur < res + data_length; cur++) {
    *cur = data++;
  }
  for (mnist_data **cur = res; cur < res + data_length; cur++) {
    mnist_data **r = res + rand()%data_length;
    mnist_data *tmp = *r;
    *r = *cur;
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

int layer_init(layer *l, unsigned size, memory *inputs) {
  int err;
  
  err = mem_init(&l->weights, inputs->length*size, sizeof(float));
  if (err) return err;
  err = mem_init(&l->biases, inputs->length, sizeof(float));
  if (err) goto err_biases_init;
  err = mem_init(&l->outputs, size, sizeof(float));
  if (err) goto err_output_init;
  err = mem_init(&l->error_term, size, sizeof(float));
  if (err) goto err_error_term_init;
  err = mem_init(&l->bias_error, size, sizeof(float));
  if (err) goto err_bias_error_init;
  err = mem_init(&l->weight_error, inputs->length*size, sizeof(float));
  if (err) goto err_weight_error_init;
  l->inputs = inputs;
  l->size = size;
  
  for (float *cur = l->weights.ptr; cur < ((float *) l->weights.ptr) + l->weights.length; cur++) {
    *cur = scaled_rand();
  }
  for (float *cur = l->biases.ptr; cur < ((float *) l->biases.ptr) + l->biases.length; cur++) {
    *cur = scaled_rand();
  }
  for (float *cur = l->error_term.ptr; cur < ((float *) l->error_term.ptr) + l->error_term.length; cur++) {
    *cur = 0;
  }
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
} network;

int network_init(network *n, memory *input, unsigned *layer_sizes) {
  int err = 0;
  
  n->layers = malloc(sizeof(layer));
  if (!n->layers) return -1;
  
  err = layer_init(n->layers, *layer_sizes, input);
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
    err = layer_init(&n->layers[n->size], *layer_sizes, &n->layers[n->size - 1].outputs);
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

/*
  stores σ(wˡaˡ⁻¹ + bˡ) in *outputs for each neuron in each layer
  stores wˡaˡ⁻¹ + bˡ in *error_term for each neuron in each layer, NOTE: this is not yet δ, this is just z
*/
void network_compute(network *n) {
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    for (unsigned id = 0; id < cur->size; id++) {
      float *output = ((float *) cur->outputs.ptr) + id;
      float *weights = ((float *) cur->weights.ptr) + id*cur->inputs->length;
      float *bias = ((float *) cur->biases.ptr) + id;
      float *error_term = ((float *) cur->error_term.ptr) + id;
      float res = fdot(cur->inputs->ptr, weights, cur->inputs->length) + *bias;
      float raised = exp(-res);
      if (isnan(raised)) {
        *output = 0;
        *error_term = 0;
      } else {
        res = 1/(1+raised);
        *output = res;
        *error_term = raised*res*res;
      }
    }
  }
}

/*
  stores ∇ₐC⊙σ′(zˡ) in *error_term for each neuron in each layer if layer l is the network's final layer
  otherwise stores ((wˡ⁺¹)ᵀδˡ⁺¹)⊙σ′(zˡ) in *error_term for each neuron in each layer
  NOTE: error_term initially stores z as computed in network_compute
*/
void network_compute_err(network *n, float *answer) {
  layer *cur = n->layers + n->size - 1;
  
  for (unsigned id = 0; id < cur->size; id++) {
    float *error_term = (float *) cur->error_term.ptr + id;
    float *bias_error = ((float *) cur->bias_error.ptr) + id;
    
    float error = *(id + (float *) cur->outputs.ptr) - answer[id];
    error *= *error_term;
    
    *error_term = error;
    *bias_error += error;
    
    float *weight_error = ((float *) cur->weight_error.ptr) + id*cur->size;
    float *c_prev_input = (float *) cur->inputs->ptr;
    float *w_end = weight_error + cur->inputs->length;
    while (weight_error < w_end) {
      *weight_error += error * (*c_prev_input);
      weight_error++;
      c_prev_input++;
    }
    //printf("%f ", error);
  }
  //printf("\n");
  cur--;
  
  for (; cur >= n->layers; cur--) {
    for (unsigned id = 0; id < cur->size; id++) {
      layer *next = cur + 1;
      float *error_term = ((float *) cur->error_term.ptr) + id;
      
      float *bias_error = ((float *) cur->bias_error.ptr) + id;
      float *next_weights = ((float *) next->weights.ptr) + id; // +id and using custom inline fdot for properly transposing this matrix
      
      float error = 0;
      float *c_error_term = (float *) next->error_term.ptr;
      float *end = c_error_term + next->size;
      while (c_error_term < end) {
        error += (*c_error_term) * (*next_weights);
        c_error_term++;
        next_weights += cur->size;
      }
      error *= *error_term;
      
      //printf("%f ", res);
      
      *error_term = error; //error_term will be used by the previous layer
      *bias_error += error;
      
      float *weight_error = ((float *) cur->weight_error.ptr) + id*cur->size;
      float *c_prev_input = (float *) cur->inputs->ptr;
      float *w_end = weight_error + cur->inputs->length;
      while (weight_error < w_end) {
        *weight_error += error * (*c_prev_input);
        weight_error++;
        c_prev_input++;
      }
    }
    //printf("\n");
  }
}

void network_update_neurons(network *n, float learning_rate, unsigned mini_batch_size) {
  float coefficient = learning_rate/((float) mini_batch_size);
  
  for (layer *cur = n->layers; cur < n->layers + n->size; cur++) {
    for (unsigned i = 0; i < cur->size; i++) {
      float *weight_error = (float *) cur->weight_error.ptr;
      float *weight = (float *) cur->weights.ptr;
      float *w_end = weight + cur->inputs->length*cur->size;
      while (weight < w_end) {
        *weight = *weight - coefficient * (*weight_error);
        *weight_error = 0;
        weight++;
        weight_error++;
      }
      
      float *bias_error = (float *) cur->bias_error.ptr;
      float *bias = (float *) cur->biases.ptr;
      float *b_end = bias + cur->size;
      while (bias < b_end) {
        *bias = *bias - coefficient * (*bias_error);
        *bias_error = 0;
        bias++;
        bias_error++;
      }
    }
  }
}

void network_train(network *n, mnist_data **data, unsigned data_length, float learning_rate, unsigned mini_batch_size) {
  float answer[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  mnist_data **cur = data;
  
  //int iterations_remaining = 462; //debug variable
  
  for (mnist_data **end = data + mini_batch_size; end <= data + data_length; end += mini_batch_size) {
    //if (!iterations_remaining--) return; //debug
    unsigned num_correct = 0;
    for (; cur < end; cur++) {
      int last_answer_index = (*cur)->label;
      answer[last_answer_index] = 1;
      n->layers[0].inputs->ptr = &(*cur)->data[0][0];
      
      /*
      for (unsigned i = 0; i < n->layers[0].inputs->length; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", *(((float *) n->layers[0].inputs->ptr) + i) >= 0.5 ? '1' : '0');
      }
      printf("\n");
      */
      
      network_compute(n);
      
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
      
      network_compute_err(n, &answer[0]);
      answer[last_answer_index] = 0;
    }
    printf("%u/%u\n", num_correct, mini_batch_size);
    network_update_neurons(n, learning_rate, mini_batch_size);
  }
}

#define PANIC(msg, code) { printf("%s failed with error code %d.\n", msg, code); exit(-1); }

int main() {
  mnist_data *data;
  unsigned data_count;
  int err = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &data, &data_count);
  if (err) PANIC("mnist_load", err);
  memory input;
  input.ptr = &data[0].data[0];
  input.length = 28*28;
  input.unit_size = sizeof(float);
  
  mnist_data **shuffled = shuffle_data(data, data_count);
  if (!shuffled) PANIC("shuffle_data", -1);
  
  network n;
  unsigned layer_sizes[] = {15, 10, 0};
  network_init(&n, &input, &layer_sizes[0]);
  
  float learning_rate = 1.0;
  unsigned mini_batch_size = 10;
  network_train(&n, shuffled, data_count, learning_rate, mini_batch_size);
  
  free(data);
  free(shuffled);
  network_destroy(&n);
  return 0;
}
