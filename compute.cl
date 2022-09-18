inline float fdot(global float *a, global float *b, size_t len) {
  float res = 0;
  global float *end = a + len;
  while (a < end) {
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
  printf("(res:%f output:%f) ", *error_term, *output); //debug
}

/*
  *error_term must initially be equal to σ′(wˡaˡ⁻¹ + bˡ)
  computes ((wˡ⁺¹)ᵀδˡ⁺¹)⊙σ′(wˡaˡ⁻¹ + bˡ), stored in *error_term
*/
kernel void finish_err_compute(global float *error_term, global float *next_weights, global float *next_error_terms, const unsigned layer_size) {
  int id = get_global_id(0);
  if (id >= (int) layer_size) return;
  error_term += id;
  next_weights += id*layer_size; // this is being properly transposed or is already transposed, right?
  next_error_terms += id;
  float res = fdot(next_weights, next_error_terms, layer_size);
  *error_term = (*error_term) * res;
  printf("%f ", *error_term);
}
