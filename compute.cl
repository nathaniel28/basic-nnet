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

inline float fdot(global float *a, global float *b, size_t len) {
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
kernel void compute_output_and_err(global float *input, const unsigned input_count, global float *output, global float *weights, global float *bias, global float *error_term, const unsigned layer_size) {
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
  printf("(res:%f output:%f) ", *error_term, *output);
}

/*
  *error_term must initially be equal to σ′(wˡaˡ⁻¹ + bˡ)
  computes ((wˡ⁺¹)ᵀδˡ⁺¹)⊙σ′(wˡaˡ⁻¹ + bˡ), stored in *error_term
*/
kernel void finish_err_compute(global float *error_term, const unsigned size, global float *next_weights, global float *next_error_terms, const unsigned next_size) {
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
  
  *error_term = (*error_term) * res;
  printf("%f ", *error_term);
}

#ifdef NO_GPU
#undef kernel
#undef global
#undef inline
#endif
