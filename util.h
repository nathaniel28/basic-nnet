#include <stdio.h>
#include <stdlib.h>

typedef struct {
  void *ptr;
  size_t length;
  size_t unit_size;
} memory;

extern int mem_init(memory *m, size_t length, size_t unit_size) {
  size_t buffer_size = length*unit_size;
  
  m->ptr = malloc(buffer_size);
  if (!m->ptr) return -1;
  
  m->length = length;
  m->unit_size = unit_size;
  
  return 0;
}

extern void mem_destroy(memory *m) {
  free(m->ptr);
}
