OBJECTS=main.o timer.o

CFLAGS=-std=gnu99 -O3 -Wall -g
LDFLAGS=-lm -lrt -lOpenCL

all: seq opencl

seq: $(OBJECTS) colorizer_seq.o
	$(CC) -o colorizer_seq $^ $(LDFLAGS)

opencl: $(OBJECTS) colorizer_opencl.o
	$(CC) -o colorizer_opencl $^ $(LDFLAGS)

clean:
	rm -rf colorizer_seq colorizer_opencl $(OBJECTS) colorizer_seq.o colorizer_opencl.o
