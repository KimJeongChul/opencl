#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
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

int mat_mul_opencl(float *A, float *B, float*C, int ROW_A, int COL_A, int COL_B) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel kernel;
  cl_int err;

  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  // Get device gpu
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);

  // Create context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create Queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Get kernel source
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
  kernel = clCreateKernel(program, "mat_mul", &err);
  CHECK_ERROR(err);

  cl_mem bufferA, bufferB, bufferC;

  // Create buffer
  bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*ROW_A*COL_A,
    NULL, &err);
  CHECK_ERROR(err);

  bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*COL_A*COL_B,
    NULL, &err);
  CHECK_ERROR(err);

  bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ROW_A*COL_B,
    NULL, &err);
  CHECK_ERROR(err);

  double start_time = get_time();

  // Write buffer
  err = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(float)*ROW_A*COL_A, A, 0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(float)*COL_A*COL_B, B, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Set kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 3, sizeof(cl_int), &ROW_A);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 4, sizeof(cl_int), &COL_A);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 5, sizeof(cl_int), &COL_B);
  CHECK_ERROR(err);

  // Set global, local size
  size_t global_size[2] = {COL_B, ROW_A};
  size_t local_size[2] = {16, 16};

  global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
  global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

  // Enquque nd range kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float)*ROW_A*COL_B, C, 0, NULL, NULL);
  CHECK_ERROR(err);

  double end_time = get_time();
  printf("Elasped Time(excl. initialization): %f sec\n", end_time - start_time);

  // Release OpenCL object
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  printf("Finished!\n");

  return 0;
}