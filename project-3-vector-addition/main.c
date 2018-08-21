#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define VECTOR_SIZE 90000
#define LOCAL_SIZE 32

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

int main() {
  // OpenCl Variables
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_mem bufferA, bufferB, bufferC;
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel kernel;
  cl_int err;
  size_t global_size = VECTOR_SIZE;
  size_t local_size = LOCAL_SIZE;

  // Create Vector A, B, C
  int *A = (int*)malloc(sizeof(int) * VECTOR_SIZE);
  int *B = (int*)malloc(sizeof(int) * VECTOR_SIZE);
  int *C = (int*)malloc(sizeof(int) * VECTOR_SIZE);

  // Initial Vector A, B
  cl_ushort idx;
  for(idx = 0; idx < VECTOR_SIZE; idx++) {
    A[idx] = rand() % 100;
    B[idx] = rand() % 100;
  }

  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  // Get GPU device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);

  // Create context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Get kernel code
  kernel_source = get_source_code("kernel.cl", &kernel_source_size);

  // Create program
  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
    &kernel_source_size, &err);
  CHECK_ERROR(err);

  // Build program
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if(err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    char *log;

    // Get program build
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
      0, NULL, &log_size);
    CHECK_ERROR(err);

    // Get build log
    log = (char*)malloc(log_size + 1);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
      log_size, log, NULL);
    CHECK_ERROR(err);

    log[log_size] = '\0';
    printf("Compiler error : \n%s\n", log);
    free(log);
    exit(0);
  }
  CHECK_ERROR(err);

  // Create kernel
  kernel = clCreateKernel(program, "vec_add", &err);
  CHECK_ERROR(err);

  // Create Buffer
  bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
  CHECK_ERROR(err);

  bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
  CHECK_ERROR(err);

  bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * VECTOR_SIZE, NULL, &err);
  CHECK_ERROR(err);

  // Create command-queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Write Buffer
  err = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(int) * VECTOR_SIZE, A, 0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(int) * VECTOR_SIZE, B, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Set Kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  CHECK_ERROR(err);

  // Execute Kernel
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Read Buffer
  err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int) * VECTOR_SIZE, C, NULL, NULL);
  CHECK_ERROR(err);

  for(idx = 0; idx < VECTOR_SIZE; idx++)
    if(idx % 10000 == 0)
      printf("%d", C[idx]);
  printf("Finished !\n");

  // Release OpenCL object
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  printf("Finished!\n");
  return 0;
}