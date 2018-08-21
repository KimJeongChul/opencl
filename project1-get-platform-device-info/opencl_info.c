#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define PLATFORM_NAME 512
#define PLATFORM_VENDOR 256

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

int main() {
  // platform variables
  cl_uint num_platforms;
  cl_platform_id *platforms;
  char platform_name[PLATFORM_NAME];
  char platform_vendor[PLATFORM_VENDOR];
  cl_int err;

  // Get number of platforms
  clGetPlatformIDs(0, NULL, &num_platforms);
  

  // Get platforms
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);
  CHECK_ERROR(err);

  // Get platforms Info
  cl_ushort idx;
  printf("Number of platforms : %d\n", num_platforms);
  for(idx = 0; idx < num_platforms; idx ++) {
    clGetPlatformInfo(platforms[idx], CL_PLATFORM_NAME, sizeof(char) * PLATFORM_NAME, platform_name, NULL);
    clGetPlatformInfo(platforms[idx], CL_PLATFORM_VENDOR, sizeof(char) * PLATFORM_VENDOR, platform_vendor, NULL);
    printf("[*] platform : %d\n", idx);
    printf("CL_PLATFORM_NAME : %s\n", platform_name);
    printf("CL_PLATFORM_VENDOR : %s\n", platform_vendor);
  }

  return 0;
}