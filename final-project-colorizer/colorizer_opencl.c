#include <CL/cl.h>
#include "colorizer.h"
#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

// OpenCL Variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
char *kernel_source;
size_t kernel_source_size;

// Kernel Variables
cl_kernel kernel_conv;
cl_kernel kernel_fc;
cl_kernel kernel_relu;
cl_kernel kernel_sigmoid;
cl_kernel kernel_fuse;
cl_kernel kernel_unsample;

// Error Variables
cl_int err;

void set_kernel_cov_arguments(cl_mem* buf_in, cl_mem* buf_out, cl_mem* buf_weight, cl_mem* buf_bias,
    int H, int W, int K, int C, int stride, size_t size) {
    /*
    __global float *in,
    __global float *out,
    __global float *weight,
    __global float *bias,
    int H, int W, int K, int C, int stride
    */
    err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), buf_in);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), buf_out);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), buf_weight);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 3, sizeof(cl_mem), buf_bias);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 4, sizeof(int), &H);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 5, sizeof(int), &W);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 6, sizeof(int), &K);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 7, sizeof(int), &C);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 8, sizeof(int), &stride);
    CHECK_ERROR(err);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_conv, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void set_kernel_fc_arguments(cl_mem* buf_in, cl_mem* buf_out, cl_mem* buf_weight, cl_mem* buf_bias,
    int K, int C, size_t size) {
    /*
    __global float *in,
    __global float *out,
    __global float *weight,
    __global float *bias,
    int K, int C
    */
    err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), buf_in);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), buf_out);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), buf_weight);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 3, sizeof(cl_mem), buf_bias);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 4, sizeof(int), &K);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 5, sizeof(int), &C);
    CHECK_ERROR(err);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_fc, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void set_kernel_relu_arguments(cl_mem* buf_inout, int CHW, size_t size) {
    /*
    __global float *inout, 
    int CHW
    */
    err = clSetKernelArg(kernel_relu, 0, sizeof(cl_mem), buf_inout);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_relu, 1, sizeof(int), &CHW);
    CHECK_ERROR(err);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_relu, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void set_kernel_sigmoid_arguments(cl_mem* buf_inout, int CHW, size_t size) {
    /*
    __global float *inout, 
    int CHW
    */
    err = clSetKernelArg(kernel_sigmoid, 0, sizeof(cl_mem), buf_inout);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_sigmoid, 1, sizeof(int), &CHW);
    CHECK_ERROR(err);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_sigmoid, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void set_kernel_fuse_arguments(cl_mem* buf_ml, cl_mem* buf_gf, cl_mem* buf_inout, size_t size) {
    /*
    __global float* ml,
    __global float* gf,
    __global float* out
    */
    err = clSetKernelArg(kernel_fuse, 0, sizeof(cl_mem), buf_ml);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fuse, 1, sizeof(cl_mem), buf_gf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fuse, 2, sizeof(cl_mem), buf_inout);
    CHECK_ERROR(err);
    //err = clSetKernelArg(kernel_fuse, 3, sizeof(int), &CHW);
    //CHECK_ERROR(err);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_fuse, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void set_kernel_upsample_arguments(cl_mem* buf_in, cl_mem* buf_out,
    int H, int W, int C) {
    /*
    __global float* in,
    __global float* out, 
    int H, int W, int C
    */
    err = clSetKernelArg(kernel_unsample, 0, sizeof(cl_mem), buf_in);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_unsample, 1, sizeof(cl_mem), buf_out);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_unsample, 2, sizeof(int), &H);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_unsample, 3, sizeof(int), &W);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_unsample, 4, sizeof(int), &C);
    CHECK_ERROR(err);
    
    size_t global_size = C * W * 2 * H * 2;
    size_t local_size = 256;
    
    global_size = (global_size + local_size - 1) / local_size * local_size;
    
    err = clEnqueueNDRangeKernel(queue, kernel_unsample, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
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

void colorizer_init() {
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
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        // Get program build
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        // Get build log
        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error : \n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    // Create kernel
    kernel_conv = clCreateKernel(program, "conv", &err);
    CHECK_ERROR(err);

    kernel_fc = clCreateKernel(program, "fc", &err);
    CHECK_ERROR(err);

    kernel_relu = clCreateKernel(program, "relu", &err);
    CHECK_ERROR(err);

    kernel_sigmoid = clCreateKernel(program, "sigmoid", &err);
    CHECK_ERROR(err);

    kernel_fuse = clCreateKernel(program, "fuse", &err);
    CHECK_ERROR(err);

    kernel_unsample = clCreateKernel(program, "unsample", &err);
    CHECK_ERROR(err);
}

void colorizer(int nimg, float *network, float *inputs, float *outputs) {
    /*
     * Low-Level Feature Network
     */
    // Create and Write ll_conv1_w buffer
    cl_mem buf_ll_conv1_w, buf_ll_conv1_b;
    float *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
    buf_ll_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64 * 1 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv1_w, CL_FALSE, 0, sizeof(float) * 64 * 1 * 3 * 3, ll_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv1_b buffer
    float *ll_conv1_b = network; network += 64;
    buf_ll_conv1_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv1_b, CL_FALSE, 0, sizeof(float) * 64, ll_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv2_w buffer
    cl_mem buf_ll_conv2_w, buf_ll_conv2_b;
    float *ll_conv2_w = network; network += 128 * 64 * 3 * 3;
    buf_ll_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128 * 64 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv2_w, CL_FALSE, 0, sizeof(float) * 128 * 64 * 3 * 3, ll_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv2_b buffer
    float *ll_conv2_b = network; network += 128;
    buf_ll_conv2_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv2_b, CL_FALSE, 0, sizeof(float) * 128, ll_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv3_w buffer
    cl_mem buf_ll_conv3_w, buf_ll_conv3_b;
    float *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
    buf_ll_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128 * 128 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv3_w, CL_FALSE, 0, sizeof(float) * 128 * 128 * 3 * 3, ll_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv3_b buffer
    float *ll_conv3_b = network; network += 128;
    buf_ll_conv3_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv3_b, CL_FALSE, 0, sizeof(float) * 128, ll_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv4_w buffer
    cl_mem buf_ll_conv4_w, buf_ll_conv4_b;
    float *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
    buf_ll_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 128 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv4_w, CL_FALSE, 0, sizeof(float) * 256 * 128 * 3 * 3, ll_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv4_b buffer
    float *ll_conv4_b = network; network += 256;
    buf_ll_conv4_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv4_b, CL_FALSE, 0, sizeof(float) * 256, ll_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv5_w buffer
    cl_mem buf_ll_conv5_w, buf_ll_conv5_b;
    float *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
    buf_ll_conv5_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 256 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv5_w, CL_FALSE, 0, sizeof(float) * 256 * 256 * 3 * 3, ll_conv5_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv5_b buffer
    float *ll_conv5_b = network; network += 256;
    buf_ll_conv5_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv5_b, CL_FALSE, 0, sizeof(float) * 256, ll_conv5_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv6_w buffer
    cl_mem buf_ll_conv6_w, buf_ll_conv6_b;
    float *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
    buf_ll_conv6_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 256 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv6_w, CL_FALSE, 0, sizeof(float) * 512 * 256 * 3 * 3, ll_conv6_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv6_b buffer
    float *ll_conv6_b = network; network += 512;
    buf_ll_conv6_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv6_b, CL_FALSE, 0, sizeof(float) * 512, ll_conv6_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Mid-Level Feature Network
     */
    // Create and Write ml_conv1_w buffer
    cl_mem buf_ml_conv1_w, buf_ml_conv1_b;
    float *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
    buf_ml_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_w, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, ml_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv1_b buffer
    float *ml_conv1_b = network; network += 512;
    buf_ml_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_b, CL_FALSE, 0, sizeof(float) * 512, ml_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv2_w buffer
    cl_mem buf_ml_conv2_w, buf_ml_conv2_b;
    float *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
    buf_ml_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv2_w, CL_FALSE, 0, sizeof(float) * 256 * 512 * 3 * 3, ml_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv2_b buffer
    float *ml_conv2_b = network; network += 256;
    buf_ml_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_b, CL_FALSE, 0, sizeof(float) * 256, ml_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Global Feature Network
     */
    // Create and Write gf_conv1_w buffer
    cl_mem buf_gf_conv1_w, buf_gf_conv1_b;
    float *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv1_w, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, gf_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv1_b buffer
    float *gf_conv1_b = network; network += 512;
    buf_gf_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv1_b, CL_FALSE, 0, sizeof(float) * 512, gf_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv2_w buffer
    cl_mem buf_gf_conv2_w, buf_gf_conv2_b;
    float *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv2_w, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, gf_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv2_b buffer
    float *gf_conv2_b = network; network += 512;
    buf_gf_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv2_b, CL_FALSE, 0, sizeof(float) * 512, gf_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv3_w buffer
    cl_mem buf_gf_conv3_w, buf_gf_conv3_b;
    float *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv3_w, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, gf_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv3_b buffer
    float *gf_conv3_b = network; network += 512;
    buf_gf_conv3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv3_b, CL_FALSE, 0, sizeof(float) * 512, gf_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);
        
    // Create and Write gf_conv4_w buffer
    cl_mem buf_gf_conv4_w, buf_gf_conv4_b;
    float *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv4_w, CL_FALSE, 0, sizeof(float) * 512 * 512 * 3 * 3, gf_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv4_b buffer
    float *gf_conv4_b = network; network += 512;
    buf_gf_conv4_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv4_b, CL_FALSE, 0, sizeof(float) * 512, gf_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Global Feature Fully Connected Layer
     */
    // Create and Write gf_fc1_w buffer
    cl_mem buf_gf_fc1_w, buf_gf_fc1_b;
    float *gf_fc1_w = network; network += 1024 * 25088;
    buf_gf_fc1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 1024 * 25088, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc1_w, CL_FALSE, 0, sizeof(float) * 1024 * 25088, gf_fc1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc1_b buffer
    float *gf_fc1_b = network; network += 1024;
    buf_gf_fc1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 1024, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc1_b, CL_FALSE, 0, sizeof(float) * 1024, gf_fc1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc2_w buffer
    cl_mem buf_gf_fc2_w, buf_gf_fc2_b;
    float *gf_fc2_w = network; network += 512 * 1024;
    buf_gf_fc2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 1024, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc2_w, CL_FALSE, 0, sizeof(float) * 512 * 1024, gf_fc2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc2_b buffer
    float *gf_fc2_b = network; network += 512;
    buf_gf_fc2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc2_b, CL_FALSE, 0, sizeof(float) * 512, gf_fc2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc3_w buffer
    cl_mem buf_gf_fc3_w, buf_gf_fc3_b;
    float *gf_fc3_w = network; network += 512 * 1024;
    buf_gf_fc3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512 * 1024, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc3_w, CL_FALSE, 0, sizeof(float) * 512 * 1024, gf_fc3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc3_b buffer
    float *gf_fc3_b = network; network += 512;
    buf_gf_fc3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc3_b, CL_FALSE, 0, sizeof(float) * 512, gf_fc3_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    /*
     * Colorization Layer
     */
    // Create and Write co_conv1_w buffer
    cl_mem buf_co_conv1_w, buf_co_conv1_b;
    float *co_conv1_w = network; network += 256 * 512 * 3 * 3;
    buf_co_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256 * 512 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv1_w, CL_FALSE, 0, sizeof(float) * 256 * 512 * 3 * 3, co_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv1_b buffer
    float *co_conv1_b = network; network += 256;
    buf_co_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 256, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv1_b, CL_FALSE, 0, sizeof(float) * 256, co_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv2_w buffer
    cl_mem buf_co_conv2_w, buf_co_conv2_b;
    float *co_conv2_w = network; network += 128 * 256 * 3 * 3;
    buf_co_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128 * 256 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv2_w, CL_FALSE, 0, sizeof(float) * 128 * 256 * 3 * 3, co_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv2_b buffer
    float *co_conv2_b = network; network += 128;
    buf_co_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 128, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv2_b, CL_FALSE, 0, sizeof(float) * 128, co_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv3_w buffer
    cl_mem buf_co_conv3_w, buf_co_conv3_b;
    float *co_conv3_w = network; network += 64 * 128 * 3 * 3;
    buf_co_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64 * 128 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv3_w, CL_FALSE, 0, sizeof(float) * 64 * 128 * 3 * 3, co_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv3_b buffer
    float *co_conv3_b = network; network += 64;
    buf_co_conv3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv3_b, CL_FALSE, 0, sizeof(float) * 64, co_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv4_w buffer
    cl_mem buf_co_conv4_w, buf_co_conv4_b;
    float *co_conv4_w = network; network += 64 * 64 * 3 * 3;
    buf_co_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64 * 64 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv4_w, CL_FALSE, 0, sizeof(float) * 64 * 64 * 3 * 3, co_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv4_b buffer
    float *co_conv4_b = network; network += 64;
    buf_co_conv4_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 64, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv4_b, CL_FALSE, 0, sizeof(float) * 64, co_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv5_w buffer
    cl_mem buf_co_conv5_w, buf_co_conv5_b;
    float *co_conv5_w = network; network += 32 * 64 * 3 * 3;
    buf_co_conv5_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 64 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv5_w, CL_FALSE, 0, sizeof(float) * 32 * 64 * 3 * 3, co_conv5_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv5_b buffer
    float *co_conv5_b = network; network += 32;
    buf_co_conv5_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv5_b, CL_FALSE, 0, sizeof(float) * 32, co_conv5_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv6_w buffer
    cl_mem buf_co_conv6_w, buf_co_conv6_b;
    float *co_conv6_w = network; network += 2 * 32 * 3 * 3;
    buf_co_conv6_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 2 * 32 * 3 * 3, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv6_w, CL_FALSE, 0, sizeof(float) * 2 * 32 * 3 * 3, co_conv6_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv6_b buffer
    float *co_conv6_b = network; network += 2;
    
    buf_co_conv6_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 2, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv6_b, CL_FALSE, 0, sizeof(float) * 2, co_conv6_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    

    /*
    * Low-Level intermediate buffer for feature maps
    */
    // Create ll_fm1 buffer
    cl_mem buf_ll_fm1, buf_ll_fm2, buf_ll_fm3, buf_ll_fm4, buf_ll_fm5, buf_ll_fm6;
    float *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    buf_ll_fm1  = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // Create ll_fm2 buffer
    float *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
    buf_ll_fm2  = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // Create ll_fm3 buffer
    float *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    buf_ll_fm3  = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm4 buffer
    float *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    buf_ll_fm4  = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm5 buffer
    float *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    buf_ll_fm5  = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm6 buffer
    float *ll_fm6 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    buf_ll_fm6  = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    /*
     * Mid-Level intermediate buffer for feature maps
    */
    // Create ml_fm1 buffer
    cl_mem buf_ml_fm1, buf_ml_fm2;
    float *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    buf_ml_fm1  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ml_fm2 buffer
    float *ml_fm2 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    buf_ml_fm2  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    /*
     * Global intermediate buffer for feature maps
    */
    // Create gf_fm1 buffer
    cl_mem buf_gf_fm1, buf_gf_fm2, buf_gf_fm3, buf_gf_fm4, buf_gf_fm5, buf_gf_fm6, buf_gf_fm7;
    float *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    buf_gf_fm1  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm2 buffer
    float *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    buf_gf_fm2  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm3 buffer
    float *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    buf_gf_fm3  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm4 buffer
    float *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    buf_gf_fm4  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm5 buffer
    float *gf_fm5 = (float*)malloc(1024 * sizeof(float));
    buf_gf_fm5  = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm6 buffer
    float *gf_fm6 = (float*)malloc(512 * sizeof(float));
    buf_gf_fm6  = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm7 buffer
    float *gf_fm7 = (float*)malloc(256 * sizeof(float));
    buf_gf_fm7  = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    /*
     * Mid-Level Global intermediate buffer for feature maps
    */
    // Create ml_gf_fused_fm buffer
    cl_mem buf_ml_gf_fused_fm;
    float *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
    buf_ml_gf_fused_fm = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    /*
     * Colorization intermediate buffer for feature maps
    */
    // Create co_fm1 buffer
    cl_mem buf_co_fm1, buf_co_fm2, buf_co_fm3, buf_co_fm4, buf_co_fm5, buf_co_fm6, buf_co_fm7;
    float *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    buf_co_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm2 buffer
    float *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
    buf_co_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm3 buffer
    float *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    buf_co_fm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm4 buffer
    float *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    buf_co_fm4 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm5 buffer
    float *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    buf_co_fm5 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm6 buffer
    float *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    buf_co_fm6 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm7 buffer
    float *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));
    buf_co_fm7 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // run network for each image
    for (int n = 0; n < nimg; ++n) {
        float *input = inputs + n * 224 * 224;
        float *output = outputs + n * 2 * 112 * 112;
        
        cl_mem buf_input, buf_output;
        buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 224 * 224, NULL, &err);
        CHECK_ERROR(err);
        
        err = clEnqueueWriteBuffer(queue, buf_input, CL_FALSE, 0, sizeof(float) * 224 * 224, input, 0, NULL, NULL);
        CHECK_ERROR(err);
        
        buf_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 2 * 112 * 112, NULL, &err);
        
        err = clEnqueueWriteBuffer(queue, buf_output, CL_FALSE, 0, sizeof(float) * 2 * 112 * 112, output, 0, NULL, NULL);
        CHECK_ERROR(err);
        
        set_kernel_cov_arguments(buf_input, buf_ll_fm1, buf_ll_conv1_w, buf_ll_conv1_b, 224, 224, 64, 1, 2, 64 * 112 * 112);
        set_kernel_relu_arguments(buf_ll_fm1, 64 * 112 * 112, 64 * 112 * 112);
        set_kernel_cov_arguments(buf_ll_fm1, buf_ll_fm2, buf_ll_conv2_w, buf_ll_conv2_b, 112, 112, 128, 64, 1, 128 * 112 * 112);
        set_kernel_relu_arguments(buf_ll_fm2, 128 * 112 * 112, 128 * 112 * 112);
        set_kernel_cov_arguments(buf_ll_fm2, buf_ll_fm3, buf_ll_conv3_w, buf_ll_conv3_b, 112, 112, 128, 128, 2, 128 * 56 * 56);
        set_kernel_relu_arguments(buf_ll_fm3, 128 * 56 * 56, 128 * 56 * 56);
        set_kernel_cov_arguments(buf_ll_fm3, buf_ll_fm4, buf_ll_conv4_w, buf_ll_conv4_b, 56, 56, 256, 128, 1, 256 * 56 * 56);
        set_kernel_relu_arguments(buf_ll_fm4, 256 * 56 * 56, 256 * 56 * 56);
        set_kernel_cov_arguments(buf_ll_fm4, buf_ll_fm5, buf_ll_conv5_w, buf_ll_conv5_b, 56, 56, 256, 256, 2, 256 * 28 * 28);
        set_kernel_relu_arguments(buf_ll_fm5, 256 * 28 * 28, 256 * 28 * 28);
        set_kernel_cov_arguments(buf_ll_fm5, buf_ll_fm6, buf_ll_conv6_w, buf_ll_conv6_b, 28, 28, 512, 256, 1, 512 * 28 * 28);
        set_kernel_relu_arguments(buf_ll_fm6, 512 * 28 * 28, 512 * 28 * 28);
        
        set_kernel_cov_arguments(buf_ll_fm6, buf_ml_fm1, buf_ml_conv1_w, buf_ml_conv1_b, 28, 28, 512, 512, 1, 512 * 28 * 28);
        set_kernel_relu_arguments(buf_ml_fm1, 512 * 28 * 28, 512 * 28 * 28);
        set_kernel_cov_arguments(buf_ml_fm1, buf_ml_fm2, ml_conv2_w, buf_ml_conv2_b, 28, 28, 256, 512, 1, 256 * 28 * 28);
        set_kernel_relu_arguments(buf_ml_fm2, 256 * 28 * 28, 256 * 28 * 28);
        
        set_kernel_cov_arguments(buf_ll_fm6, buf_gf_fm1, buf_gf_conv1_w, buf_gf_conv1_b, 28, 28, 512, 512, 2, 512 * 14 * 14);
        set_kernel_relu_arguments(buf_gf_fm1, 512 * 14 * 14, 512 * 14 * 14);
        set_kernel_cov_arguments(buf_gf_fm1, buf_gf_fm2, buf_gf_conv2_w, buf_gf_conv2_b, 14, 14, 512, 512, 1, 512 * 14 * 14);
        set_kernel_relu_arguments(buf_gf_fm2, 512 * 14 * 14, 512 * 14 * 14);
        set_kernel_cov_arguments(buf_gf_fm2, buf_gf_fm3, buf_gf_conv3_w, buf_gf_conv3_b, 14, 14, 512, 512, 2, 512 * 7 * 7);
        set_kernel_relu_arguments(buf_gf_fm3, 512 * 7 * 7, 512 * 7 * 7);
        set_kernel_cov_arguments(buf_gf_fm3, buf_gf_fm4, buf_gf_conv4_w, buf_gf_conv4_b, 7, 7, 512, 512, 1, 512 * 7 * 7);
        set_kernel_relu_arguments(buf_gf_fm4, 512 * 7 * 7, 512 * 7 * 7);
        set_kernel_fc_arguments(buf_gf_fm4, buf_gf_fm5, buf_gf_fc1_w, buf_gf_fc1_b, 1024, 25088, 1024);
        set_kernel_relu_arguments(buf_gf_fm5, 1024, 1024);
        set_kernel_fc_arguments(buf_gf_fm5, buf_gf_fm6, buf_gf_fc2_w, buf_gf_fc2_b, 512, 1024, 512);
        set_kernel_relu_arguments(buf_gf_fm6, 512, 512);
        set_kernel_fc_arguments(buf_gf_fm6, buf_gf_fm7, buf_gf_fc3_w, buf_gf_fc3_b, 256, 512, 256);
        set_kernel_relu_arguments(buf_gf_fm7, 256, 256);
            
        set_kernel_fuse_arguments(buf_ml_fm2, buf_gf_fm7, buf_ml_gf_fused_fm, 512 * 28 * 28);

        set_kernel_cov_arguments(buf_ml_gf_fused_fm, buf_co_fm1, buf_co_conv1_w, buf_co_conv1_b, 28, 28, 256, 512, 1, 256 * 28 * 28);
        set_kernel_relu_arguments(buf_co_fm1, 256 * 28 * 28, 256 * 28 * 28);
        set_kernel_cov_arguments(buf_co_fm1, buf_co_fm2, buf_co_conv2_w, buf_co_conv2_b, 28, 28, 128, 256, 1, 128 * 28 * 28);
        set_kernel_relu_arguments(buf_co_fm2, 128 * 28 * 28, 128 * 28 * 28);
        set_kernel_upsample_arguments(buf_co_fm2, buf_co_fm3, 28, 28, 128);
        set_kernel_cov_arguments(buf_co_fm3, buf_co_fm4, buf_co_conv3_w, buf_co_conv3_b, 56, 56, 64, 128, 1, 64 * 56 * 56);
        set_kernel_relu_arguments(buf_co_fm4, 64 * 56 * 56, 64 * 56 * 56);
        set_kernel_cov_arguments(buf_co_fm4, buf_co_fm5, buf_co_conv4_w, buf_co_conv4_b, 56, 56, 64, 64, 1, 64 * 56 * 56);
        set_kernel_relu_arguments(buf_co_fm5, 64 * 56 * 56, 64 * 56 * 56);
        set_kernel_upsample_arguments(buf_co_fm5, buf_co_fm6, 56, 56, 64);
        set_kernel_cov_arguments(buf_co_fm6, buf_co_fm7, buf_co_conv5_w, buf_co_conv5_b, 112, 112, 32, 64, 1, 32 * 112 * 112);
        set_kernel_relu_arguments(buf_co_fm7, 32 * 112 * 112, 32 * 112 * 112);
        set_kernel_cov_arguments(buf_co_fm7, buf_output, buf_co_conv6_w, buf_co_conv6_b, 112, 112, 2, 32, 1, 2 * 112 * 112);
        set_kernel_sigmoid_arguments(buf_output, 2 * 112 * 112, 2 * 112 * 112);
    }

    // Release OpenCL object
    clReleaseMemObject(buf_ll_conv1_w);
    clReleaseMemObject(buf_ll_conv1_b);
    clReleaseMemObject(buf_ll_conv2_w);
    clReleaseMemObject(buf_ll_conv2_b);
    clReleaseMemObject(buf_ll_conv3_w);
    clReleaseMemObject(buf_ll_conv3_b);
    clReleaseMemObject(buf_ll_conv4_w);
    clReleaseMemObject(buf_ll_conv4_b);
    clReleaseMemObject(buf_ll_conv5_w);
    clReleaseMemObject(buf_ll_conv5_b);
    clReleaseMemObject(buf_ll_conv6_w);
    clReleaseMemObject(buf_ll_conv6_b);

    clReleaseMemObject(buf_ml_conv1_w);
    clReleaseMemObject(buf_ml_conv1_b);
    clReleaseMemObject(buf_ml_conv2_w);
    clReleaseMemObject(buf_ml_conv2_b);

    clReleaseMemObject(buf_gf_conv1_w);
    clReleaseMemObject(buf_gf_conv1_b);
    clReleaseMemObject(buf_gf_conv2_w);
    clReleaseMemObject(buf_gf_conv2_b);
    clReleaseMemObject(buf_gf_conv3_w);
    clReleaseMemObject(buf_gf_conv3_b);
    clReleaseMemObject(buf_gf_conv4_w);
    clReleaseMemObject(buf_gf_conv4_b);

    clReleaseMemObject(buf_gf_fc1_w);
    clReleaseMemObject(buf_gf_fc1_b);
    clReleaseMemObject(buf_gf_fc2_w);
    clReleaseMemObject(buf_gf_fc2_b);
    clReleaseMemObject(buf_gf_fc3_w);
    clReleaseMemObject(buf_gf_fc3_b);

    clReleaseMemObject(buf_co_conv1_w);
    clReleaseMemObject(buf_co_conv1_b);
    clReleaseMemObject(buf_co_conv2_w);
    clReleaseMemObject(buf_co_conv2_b);
    clReleaseMemObject(buf_co_conv3_w);
    clReleaseMemObject(buf_co_conv3_b);
    clReleaseMemObject(buf_co_conv4_w);
    clReleaseMemObject(buf_co_conv4_b);
    clReleaseMemObject(buf_co_conv5_w);
    clReleaseMemObject(buf_co_conv5_b);
    clReleaseMemObject(buf_co_conv6_w);
    clReleaseMemObject(buf_co_conv6_b);

    clReleaseMemObject(buf_ll_fm1);
    clReleaseMemObject(buf_ll_fm2);
    clReleaseMemObject(buf_ll_fm3);
    clReleaseMemObject(buf_ll_fm4);
    clReleaseMemObject(buf_ll_fm5);
    clReleaseMemObject(buf_ll_fm6);

    clReleaseMemObject(buf_ml_fm1);
    clReleaseMemObject(buf_ml_fm2);

    clReleaseMemObject(buf_gf_fm1);
    clReleaseMemObject(buf_gf_fm2);
    clReleaseMemObject(buf_gf_fm3);
    clReleaseMemObject(buf_gf_fm4);
    clReleaseMemObject(buf_gf_fm5);
    clReleaseMemObject(buf_gf_fm6);
    clReleaseMemObject(buf_gf_fm7);

    clReleaseMemObject(buf_ml_gf_fused_fm);

    clReleaseMemObject(buf_co_fm1);
    clReleaseMemObject(buf_co_fm2);
    clReleaseMemObject(buf_co_fm3);
    clReleaseMemObject(buf_co_fm4);
    clReleaseMemObject(buf_co_fm5);
    clReleaseMemObject(buf_co_fm6);
    clReleaseMemObject(buf_co_fm7);

    clReleaseKernel(kernel_conv);
    clReleaseKernel(kernel_fc);
    clReleaseKernel(kernel_relu);
    clReleaseKernel(kernel_sigmoid);
    clReleaseKernel(kernel_fuse);
    clReleaseKernel(kernel_unsample);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}