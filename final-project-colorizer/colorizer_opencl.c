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
    int H, int W, int K, int C, int stride) {
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
}

void set_kernel_fc_arguments(cl_mem* buf_in, cl_mem* buf_out, cl_mem* buf_weight, cl_mem* buf_bias,
    int K, int C) {
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
}

void set_kernel_relu_arguments(cl_mem* buf_inout, int CHW) {
    /*
    __global float *inout, 
    int CHW
    */
    err = clSetKernelArg(kernel_relu, 0, sizeof(cl_mem), buf_inout);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_relu, 1, sizeof(int), &CHW);
    CHECK_ERROR(err);
}

void set_kernel_sigmoid_arguments(cl_mem* buf_inout, int CHW) {
    /*
    __global float *inout, 
    int CHW
    */
    err = clSetKernelArg(kernel_sigmoid, 0, sizeof(cl_mem), buf_inout);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_sigmoid, 1, sizeof(int), &CHW);
    CHECK_ERROR(err);
}

void set_kernel_fuse_arguments(cl_mem* buf_ml, cl_mem* buf_gf, cl_mem* buf_inout) {
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
    err = clSetKernelArg(kernel_fuse, 3, sizeof(int), &CHW);
    CHECK_ERROR(err);
}

void set_kernel_unsample_arguments(cl_mem* buf_in, cl_mem* buf_out,
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
    cl_mem input, dest;
    printf(sizeof(network))
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    /*
     * Low-Level Feature Network
     */
    // Create and Write ll_conv1_w buffer
    float *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
    buf_ll_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv1_w, CL_FALSE, 0, sizeof(network), ll_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv1_b buffer
    float *ll_conv1_b = network; network += 64;
    buf_ll_conv1_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv1_b, CL_FALSE, 0, sizeof(network), ll_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv2_w buffer
    float *ll_conv2_w = network; network += 64 * 1 * 3 * 3;
    buf_ll_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv2_w, CL_FALSE, 0, sizeof(network), ll_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv2_b buffer
    float *ll_conv2_b = network; network += 64;
    buf_ll_conv2_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv2_b, CL_FALSE, 0, sizeof(network), ll_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv3_w buffer
    float *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
    buf_ll_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv3_w, CL_FALSE, 0, sizeof(network), ll_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv3_b buffer
    float *ll_conv3_b = network; network += 128;
    buf_ll_conv3_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv3_b, CL_FALSE, 0, sizeof(network), ll_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv4_w buffer
    float *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
    buf_ll_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv4_w, CL_FALSE, 0, sizeof(network), ll_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv4_b buffer
    float *ll_conv4_b = network; network += 256;
    buf_ll_conv4_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv4_b, CL_FALSE, 0, sizeof(network), ll_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv5_w buffer
    float *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
    buf_ll_conv5_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv5_w, CL_FALSE, 0, sizeof(network), ll_conv5_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv5_b buffer
    float *ll_conv5_b = network; network += 256;
    buf_ll_conv5_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv5_b, CL_FALSE, 0, sizeof(network), ll_conv5_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv6_w buffer
    float *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
    buf_ll_conv6_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv6_w, CL_FALSE, 0, sizeof(network), ll_conv6_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ll_conv6_b buffer
    float *ll_conv6_b = network; network += 512;
    buf_ll_conv6_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ll_conv6_b, CL_FALSE, 0, sizeof(network), ll_conv6_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Mid-Level Feature Network
     */
    // Create and Write ml_conv1_w buffer
    float *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
    buf_ml_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_w, CL_FALSE, 0, sizeof(network), ml_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv1_b buffer
    float *ml_conv1_b = network; network += 512;
    buf_ml_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_b, CL_FALSE, 0, sizeof(network), ml_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv2_w buffer
    float *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
    buf_ml_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv2_w, CL_FALSE, 0, sizeof(network), ml_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write ml_conv2_b buffer
    float *ml_conv2_b = network; network += 256;
    buf_ml_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_ml_conv1_b, CL_FALSE, 0, sizeof(network), ml_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Global Feature Network
     */
    // Create and Write gf_conv1_w buffer
    float *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv1_w, CL_FALSE, 0, sizeof(network), gf_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv1_b buffer
    float *gf_conv1_b = network; network += 512;
    buf_gf_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv1_b, CL_FALSE, 0, sizeof(network), gf_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv2_w buffer
    float *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv2_w, CL_FALSE, 0, sizeof(network), gf_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv2_b buffer
    float *gf_conv2_b = network; network += 512;
    buf_gf_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv2_b, CL_FALSE, 0, sizeof(network), gf_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv3_w buffer
    float *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv3_w, CL_FALSE, 0, sizeof(network), gf_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv3_b buffer
    float *gf_conv3_b = network; network += 512;
    buf_gf_conv3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv3_b, CL_FALSE, 0, sizeof(network), gf_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);
        
    // Create and Write gf_conv4_w buffer
    float *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
    buf_gf_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv4_w, CL_FALSE, 0, sizeof(network), gf_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_conv4_b buffer
    float *gf_conv4_b = network; network += 512;
    buf_gf_conv3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_conv4_b, CL_FALSE, 0, sizeof(network), gf_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);


    /*
     * Global Feature Fully Connected Layer
     */
    // Create and Write gf_fc1_w buffer
    float *gf_fc1_w = network; network += 1024 * 25088;
    buf_gf_fc1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc1_w, CL_FALSE, 0, sizeof(network), gf_fc1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc1_b buffer
    float *gf_fc1_b = network; network += 1024;
    buf_gf_fc1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc1_b, CL_FALSE, 0, sizeof(network), gf_fc1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc2_w buffer
    float *gf_fc2_w = network; network += 512 * 1024;
    buf_gf_fc2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc2_w, CL_FALSE, 0, sizeof(network), gf_fc2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc2_b buffer
    float *gf_fc2_b = network; network += 512;
    buf_gf_fc2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc2_b, CL_FALSE, 0, sizeof(network), gf_fc2_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc3_w buffer
    float *gf_fc3_w = network; network += 512 * 1024;
    buf_gf_fc3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc3_w, CL_FALSE, 0, sizeof(network), gf_fc3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write gf_fc3_b buffer
    float *gf_fc3_b = network; network += 512;
    buf_gf_fc3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_gf_fc3_b, CL_FALSE, 0, sizeof(network), gf_fc3_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    /*
     * Colorization Layer
     */
    // Create and Write co_conv1_w buffer
    float *co_conv1_w = network; network += 256 * 512 * 3 * 3;
    buf_co_conv1_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv1_w, CL_FALSE, 0, sizeof(network), co_conv1_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv1_b buffer
    float *co_conv1_b = network; network += 256;
    buf_co_conv1_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv1_b, CL_FALSE, 0, sizeof(network), co_conv1_b, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv2_w buffer
    float *co_conv2_w = network; network += 128 * 256 * 3 * 3;
    buf_co_conv2_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv2_w, CL_FALSE, 0, sizeof(network), co_conv2_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv2_b buffer
    float *co_conv2_b = network; network += 128;
    buf_co_conv2_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv2_b, CL_FALSE, 0, sizeof(network), co_conv2_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv3_w buffer
    float *co_conv3_w = network; network += 64 * 128 * 3 * 3;
    buf_co_conv3_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv3_w, CL_FALSE, 0, sizeof(network), co_conv3_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv3_b buffer
    float *co_conv3_b = network; network += 64;
    buf_co_conv3_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv3_b, CL_FALSE, 0, sizeof(network), co_conv3_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv4_w buffer
    float *co_conv4_w = network; network += 64 * 64 * 3 * 3;
    buf_co_conv4_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv4_w, CL_FALSE, 0, sizeof(network), co_conv4_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv4_b buffer
    float *co_conv4_b = network; network += 64;
    buf_co_conv4_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv4_b, CL_FALSE, 0, sizeof(network), co_conv4_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv5_w buffer
    float *co_conv5_w = network; network += 32 * 64 * 3 * 3;
    buf_co_conv5_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv5_w, CL_FALSE, 0, sizeof(network), co_conv5_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv5_b buffer
    float *co_conv5_b = network; network += 32;
    buf_co_conv5_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv5_b, CL_FALSE, 0, sizeof(network), co_conv5_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Create and Write co_conv4_w buffer
    float *co_conv6_w = network; network += 2 * 32 * 3 * 3;
    buf_co_conv6_w = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(network), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv6_w, CL_FALSE, 0, sizeof(network), co_conv6_w, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create and Write co_conv4_b buffer
    float *co_conv6_b = network; network += 2;
    
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_co_conv6_b, CL_FALSE, 0, sizeof(network), co_conv6_b, 0, NULL, NULL);
    CHECK_ERROR(err);
    

    /*
    * Low-Level intermediate buffer for feature maps
    */
    // Create ll_fm1 buffer
    float *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    buf_ll_fm1  = clCreateBuffer(context, CL_MEM_READ_ONLY, 64 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // Create ll_fm2 buffer
    float *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
    buf_ll_fm2  = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // Create ll_fm3 buffer
    float *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    buf_ll_fm3  = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm4 buffer
    float *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    buf_ll_fm4  = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm5 buffer
    float *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    buf_ll_fm5  = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ll_fm6 buffer
    float *ll_fm6 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    buf_ll_fm6  = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    /*
     * Mid-Level intermediate buffer for feature maps
    */
    // Create ml_fm1 buffer
    float *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    buf_ml_fm1  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create ml_fm2 buffer
    float *ml_fm2 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    buf_ml_fm2  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    /*
     * Global intermediate buffer for feature maps
    */
    // Create gf_fm1 buffer
    float *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    buf_gf_fm1  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 14 * 14 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm2 buffer
    float *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    buf_gf_fm2  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 14 * 14 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm3 buffer
    float *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    buf_gf_fm3  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 7 * 7 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm4 buffer
    float *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    buf_gf_fm4  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * 7 * 7 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm5 buffer
    float *gf_fm5 = (float*)malloc(1024 * sizeof(float));
    buf_gf_fm5  = clCreateBuffer(context, CL_MEM_READ_ONLY, (1024 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm6 buffer
    float *gf_fm6 = (float*)malloc(512 * sizeof(float));
    buf_gf_fm6  = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create gf_fm7 buffer
    float *gf_fm7 = (float*)malloc(256 * sizeof(float));
    buf_gf_fm7  = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    /*
     * Mid-Level Global intermediate buffer for feature maps
    */
    // Create ml_gf_fused_fm buffer
    float *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
    
    
    /*
     * Colorization intermediate buffer for feature maps
    */
    // Create co_fm1 buffer
    float *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    buf_co_fm1 = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm2 buffer
    float *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
    buf_co_fm2 = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 28 * 28 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm3 buffer
    float *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    buf_co_fm3 = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm4 buffer
    float *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    buf_co_fm4 = clCreateBuffer(context, CL_MEM_READ_ONLY, 64 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm5 buffer
    float *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    buf_co_fm5 = clCreateBuffer(context, CL_MEM_READ_ONLY, 64 * 56 * 56 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm6 buffer
    float *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    buf_co_fm6 = clCreateBuffer(context, CL_MEM_READ_ONLY, 64 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // Create co_fm7 buffer
    float *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));
    buf_co_fm7 = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * 112 * 112 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    // run network for each image
    for (int n = 0; n < nimg; ++n) {
        float *input = inputs + n * 224 * 224;
        float *output = outputs + n * 2 * 112 * 112;
        conv(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2);
        relu(ll_fm1, 64 * 112 * 112);
        conv(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1);
        relu(ll_fm2, 128 * 112 * 112);
        conv(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2);
        relu(ll_fm3, 128 * 56 * 56);
        conv(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1);
        relu(ll_fm4, 256 * 56 * 56);
        conv(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2);
        relu(ll_fm5, 256 * 28 * 28);
        conv(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1);
        relu(ll_fm6, 512 * 28 * 28);

        conv(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1);
        relu(ml_fm1, 512 * 28 * 28);
        conv(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1);
        relu(ml_fm2, 256 * 28 * 28);

        conv(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2);
        relu(gf_fm1, 512 * 14 * 14);
        conv(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1);
        relu(gf_fm2, 512 * 14 * 14);
        conv(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2);
        relu(gf_fm3, 512 * 7 * 7);
        conv(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1);
        relu(gf_fm4, 512 * 7 * 7);
        fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088);
        relu(gf_fm5, 1024);
        fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024);
        relu(gf_fm6, 512);
        fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512);
        relu(gf_fm7, 256);

        fuse(ml_fm2, gf_fm7, ml_gf_fused_fm);

        conv(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1);
        relu(co_fm1, 256 * 28 * 28);
        conv(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1);
        relu(co_fm2, 128 * 28 * 28);
        upsample(co_fm2, co_fm3, 28, 28, 128);
        conv(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1);
        relu(co_fm4, 64 * 56 * 56);
        conv(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1);
        relu(co_fm5, 64 * 56 * 56);
        upsample(co_fm5, co_fm6, 56, 56, 64);
        conv(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1);
        relu(co_fm7, 32 * 112 * 112);
        conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1);
        sigmoid(output, 2 * 112 * 112);
    }
}