#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 16777216;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double reduction_seq(int *array, int N);
double reduction_opencl(int *array, int N);

int main() {
  int *array = (int*)malloc(sizeof(int) * N);
  int i;
  double ans_seq, ans_opencl;

  for (i = 0; i < N; i++) {
    array[i] = rand() % 100;
  }

  printf("Sequential version...\n");
  ans_seq = reduction_seq(array, N);
  printf("Average: %f\n", ans_seq);

  printf("OpenCL version...\n");
  ans_opencl = reduction_opencl(array, N);
  printf("Average: %f\n", ans_opencl);

  free(array);
  return 0;
}

double reduction_seq(int *array, int N) {
  int sum = 0;
  int i;
  double start_time, end_time;
  start_time = get_time();
  for (i = 0; i < N; i++) {
    sum += array[i];
  }
  end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time - start_time);
  return (double)sum / N;
}

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

double reduction_opencl(int *array, int N) {
  // TODO
}