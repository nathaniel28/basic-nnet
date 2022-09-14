#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

typedef struct {
  void *ptr;
  size_t length;
  size_t unit_size;
  cl_mem buf;
} memory;

extern int mem_init(memory *m, size_t length, size_t unit_size, cl_mem_flags flags, cl_context context) {
  size_t buffer_size = length*unit_size;
  
  m->ptr = malloc(buffer_size);
  if (!m->ptr) return CL_OUT_OF_HOST_MEMORY;
  
  int err;
  m->buf = clCreateBuffer(context, flags|CL_MEM_USE_HOST_PTR, buffer_size, m->ptr, &err);
  if (err) {
    free(m->ptr);
    return err;
  }
  
  m->length = length;
  m->unit_size = unit_size;
  
  return 0;
}

extern void mem_destroy(memory *m) {
  clReleaseMemObject(m->buf);
  free(m->ptr);
}

extern int mem_write_buffer(memory *m, cl_command_queue commands, cl_bool blocking) {
  return clEnqueueWriteBuffer(commands, m->buf, blocking, 0, m->length*m->unit_size, m->ptr, 0, NULL, NULL);
}

extern int mem_read_buffer(memory *m, cl_command_queue commands, cl_bool blocking) {
  return clEnqueueReadBuffer(commands, m->buf, blocking, 0, m->length*m->unit_size, m->ptr, 0, NULL, NULL);
}
