
CFLAGS=-g -Wall -Wextra
LIBS=-lm -lOpenCL

all::

all:: network

network: network.c
	$(CC) $(CFLAGS) -o $@ network.c $(LIBS)
