#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define PLATFORM_NAME 256
#define PLATFORM_VENDOR 256
#define DEVICE_NAME 256


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

  // device variables
  cl_uint num_devices;
  cl_device_id *devices;
  char device_name[DEVICE_NAME];
  cl_device_type device_type;
  cl_uint device_max_compute_units;
  size_t device_max_work_group_size;
  cl_ulong device_global_mem_size;
  cl_ulong device_local_mem_size;
  cl_ulong device_max_group_size;
  cl_ulong device_max_mem_alloc_size;
  cl_bool device_host_unified_memory;


  // error variable
  cl_int err;

  // Get number of platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  CHECK_ERROR(err);
  printf("[*] Number of platforms : %d\n", num_platforms);

  // Get platforms
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  CHECK_ERROR(err);

  // Get platforms Info
  cl_ushort platform_idx;
  for(platform_idx = 0; platform_idx < num_platforms; platform_idx ++) {
    err = clGetPlatformInfo(platforms[platform_idx], CL_PLATFORM_NAME, sizeof(char) * PLATFORM_NAME, platform_name, NULL);
    CHECK_ERROR(err);
    err = clGetPlatformInfo(platforms[platform_idx], CL_PLATFORM_VENDOR, sizeof(char) * PLATFORM_VENDOR, platform_vendor, NULL);
    CHECK_ERROR(err);
    printf("[-] platform : %d\n", platform_idx);
    printf("CL_PLATFORM_NAME : %s\n", platform_name);
    printf("CL_PLATFORM_VENDOR : %s\n\n", platform_vendor);

    // Get number of devices
    err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    CHECK_ERROR(err);

    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    CHECK_ERROR(err);
    printf("[-] Number of devices : %d\n", num_devices);

  	// Get devices info
    cl_ushort device_idx;
  	for(device_idx = 0; device_idx < num_devices; device_idx ++) {
  	  clGetDeviceInfo(devices[device_idx], CL_DEVICE_TYPE, sizeof(cl_device_type) , &device_type, NULL);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_NAME, sizeof(char) * DEVICE_NAME, device_name, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &device_max_compute_units, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &device_max_group_size, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_max_group_size, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem_size, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device_max_mem_alloc_size, NULL);
  	  CHECK_ERROR(err);
  	  err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_ulong), &device_host_unified_memory, NULL);
  	  CHECK_ERROR(err);
  	  printf("[*] device : %d\n", device_idx);
  	  if (device_type & CL_DEVICE_TYPE_CPU)
  	  	printf("CL_DEVICE_TYPE : CL_DEVICE_TYPE_CPU\n");
  	  if (device_type & CL_DEVICE_TYPE_GPU)
  	  	printf("CL_DEVICE_TYPE : CL_DEVICE_TYPE_GPU\n");
      printf("CL_DEVICE_NAME : %s\n", device_name);
      printf("CL_DEVICE_MAX_COMPUTE_UNITS : %d\n", device_max_compute_units);
      printf("CL_DEVICE_MAX_WORK_GROUP_SIZE : %d\n", device_max_work_group_size);
      printf("CL_DEVICE_GLOBAL_MEM_SIZE : %lu\n", device_max_group_size);
      printf("CL_DEVICE_LOCAL_MEM_SIZE : %lu\n", device_local_mem_size);
      printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE : %lu\n", device_max_mem_alloc_size);
      printf("CL_DEVICE_HOST_UNIFIED_MEMORY : %lu\n\n", device_host_unified_memory);
    }

  }

  
  return 0;
}