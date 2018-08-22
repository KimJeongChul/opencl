#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include "bmpfuncs.h"

static float theta = 3.14159/6;

void rotate(float *input_image, float *output_image, int image_width, int image_height,
            float sin_theta, float cos_theta);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <src file> <dest file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  float sin_theta = sinf(theta);
  float cos_theta = cosf(theta);

  int image_width, image_height;
  float *input_image = readImage(argv[1], &image_width, &image_height);
  float *output_image = (float*)malloc(sizeof(float) * image_width * image_height);
  rotate(input_image, output_image, image_width, image_height, sin_theta, cos_theta);
  storeImage(output_image, argv[2], image_height, image_width, argv[1]);
  return 0;
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

void rotate(float *input_image, float *output_image, int image_width, int image_height,
            float sin_theta, float cos_theta) {
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
  kernel = clCreateKernel(program, "img_rotate", &err);
  CHECK_ERROR(err);

  size_t image_size = sizeof(float) * image_width * image_height;
  cl_mem src, dest;

  // Create buffer
  src = clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, &err);
  CHECK_ERROR(err);
  dest = clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, &err);
  CHECK_ERROR(err);

  // Write buffer
  err = clEnqueueWriteBuffer(queue, src, CL_FALSE, 0, image_size, input_image, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Set kernel arguments.
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &image_width);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_int), &image_height);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 4, sizeof(cl_float), &sin_theta);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 5, sizeof(cl_float), &cos_theta);
  CHECK_ERROR(err);

  size_t global_size[2] = {image_width, image_height};
  size_t local_size[2] = {16, 16};

  global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
  global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

  // Enquque nd range kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Read buffer
  err = clEnqueueReadBuffer(queue, dest, CL_TRUE, 0, image_size, output_image, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Release OpenCL object
  clReleaseMemObject(src);
  clReleaseMemObject(dest);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
