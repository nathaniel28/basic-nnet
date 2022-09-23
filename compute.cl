#ifdef NO_GPU
#include <stdio.h>
#include <math.h>
#define kernel extern
#define global
#define inline
int fn_global_id;
int get_global_id(int _) {
  fn_global_id++;
  return fn_global_id - 1;
}
#endif

static inline float fdot(global float *a, global float *b, size_t len) {
  float res = 0;
  global float *end = a + len;
  while (a < end) {
    //printf("[[%f %f]] ", *a, *b);
    res += (*a++) * (*b++);
  }
  return res;
}

/*
  computes σ(wˡaˡ⁻¹ + bˡ) and σ′(wˡaˡ⁻¹ + bˡ)
  stored in *output and *error_term, respectively
*/
kernel void compute_output_and_err_part(global float *input, const unsigned input_count, global float *output, global float *weights, global float *bias, global float *error_term, const unsigned layer_size) {
  int id = get_global_id(0);
  if (id >= (int) layer_size) return;
  output += id;
  weights += id*input_count;
  bias += id;
  error_term += id;
  float res = fdot(input, weights, input_count) + *bias;
  float raised = exp(-res);
  res = 1/(1+raised);
  *output = res;
  *error_term = raised*res*res;
  //printf("(res:%f output:%f) ", *error_term, *output);
}

/*
  *error_term must initially be equal to σ′(wˡaˡ⁻¹ + bˡ)
  computes ((wˡ⁺¹)ᵀδˡ⁺¹)⊙σ′(wˡaˡ⁻¹ + bˡ), stored in *error_term
  adds above to *accumulated_error
*/
kernel void backpropagate_err(global float *error_term, global float *accumulated_error, const unsigned size, global float *next_weights, global float *next_error_terms, const unsigned next_size) {
  int id = get_global_id(0);
  if (id >= (int) size) return;
  error_term += id;
  next_weights += id; // +id and using custom inline fdot for properly transposing this matrix
  
  float res = 0;
  global float *end = next_error_terms + next_size;
  while (next_error_terms < end) {
    res += (*next_error_terms) * (*next_weights);
    next_error_terms++;
    next_weights += size;
  }
  res *= *error_term;
  
  *error_term = res; // TODO: error_term is not used after this, only accumulated error is (as of now). remove this write operation? If so, change error_term's name to something like "sigmoid prime output" or something as that would be it's only purpose. If so, also update documentation above it's declairation in the layer struct too!
  *accumulated_error += res;
  //printf("%f ", res);
}


/*
  TODO: description
  ∂C/∂bˡⱼ=δˡⱼ
  ∂C/∂wˡⱼₖ=aˡ⁻¹ₖδˡⱼ
*/
kernel void update_neurons(global float *weights, global float *bias, global float *accumulated_error, const unsigned size, global float *prev_output, const unsigned prev_size, const float learning_rate, const unsigned mini_batch_size) {
  int id = get_global_id(0);
  if (id >= (int) size) return;
  
  float coefficient = learning_rate/((float) mini_batch_size);
  
  global float *end = accumulated_error + size;
  while (accumulated_error < end) {
    global float *w_end = weights + prev_size;
    while (weights < w_end) {
      *weights = *weights - coefficient * (*prev_output) * (*accumulated_error);
      weights++;
      prev_output++;
    }
    prev_output -= prev_size;
    
    *bias = *bias - coefficient * (*accumulated_error);
    bias++;
    accumulated_error++;
  }
}

#ifdef NO_GPU
#undef kernel
#undef global
#undef inline
#endif
