#include "colorizer.h"

#include <mpi.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_DEV 4
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

cl_platform_id platform;
cl_device_id device[MAX_DEV];
cl_context context;
cl_command_queue command_queue[MAX_DEV];
cl_program program;
char* kernel_src;
size_t kernel_src_len;
int ndev;

cl_kernel kernel_conv[MAX_DEV];
cl_kernel kernel_conv_relu[MAX_DEV];
cl_kernel kernel_fc[MAX_DEV];
cl_kernel kernel_fuse1[MAX_DEV];
cl_kernel kernel_fuse2[MAX_DEV];
cl_kernel kernel_upsample[MAX_DEV];

static void conv(cl_mem in, cl_mem out, cl_mem weight, cl_mem bias, int H, int W, int K, int C, int stride, int dev_num) {
    int HOUT = H / stride, WOUT = W / stride;

	clSetKernelArg(kernel_conv[dev_num], 0, sizeof(cl_mem), (void*) &in);
	clSetKernelArg(kernel_conv[dev_num], 1, sizeof(cl_mem), (void*) &out);
	clSetKernelArg(kernel_conv[dev_num], 2, sizeof(cl_mem), (void*) &weight);
	clSetKernelArg(kernel_conv[dev_num], 3, sizeof(cl_mem), (void*) &bias);
	clSetKernelArg(kernel_conv[dev_num], 4, sizeof(cl_int), (void*) &H);
	clSetKernelArg(kernel_conv[dev_num], 5, sizeof(cl_int), (void*) &W);
	clSetKernelArg(kernel_conv[dev_num], 6, sizeof(cl_int), (void*) &C);
	clSetKernelArg(kernel_conv[dev_num], 7, sizeof(cl_int), (void*) &stride);

	size_t global[3] = { WOUT, HOUT, K };
	size_t local[3] = { 16, 16, 1 };

	clEnqueueNDRangeKernel(command_queue[dev_num], kernel_conv[dev_num], 3, NULL, global, local, 0, NULL, NULL);
}

static void conv_relu(cl_mem in, cl_mem out, cl_mem weight, cl_mem bias, int H, int W, int K, int C, int stride, int dev_num) {
    int HOUT = H / stride, WOUT = W / stride;

    clSetKernelArg(kernel_conv_relu[dev_num], 0, sizeof(cl_mem), (void*) &in);
    clSetKernelArg(kernel_conv_relu[dev_num], 1, sizeof(cl_mem), (void*) &out);
    clSetKernelArg(kernel_conv_relu[dev_num], 2, sizeof(cl_mem), (void*) &weight);
    clSetKernelArg(kernel_conv_relu[dev_num], 3, sizeof(cl_mem), (void*) &bias);
    clSetKernelArg(kernel_conv_relu[dev_num], 4, sizeof(cl_int), (void*) &H);
    clSetKernelArg(kernel_conv_relu[dev_num], 5, sizeof(cl_int), (void*) &W);
    clSetKernelArg(kernel_conv_relu[dev_num], 6, sizeof(cl_int), (void*) &C);
    clSetKernelArg(kernel_conv_relu[dev_num], 7, sizeof(cl_int), (void*) &stride);

    size_t global[3] = { WOUT, HOUT, K };
    size_t local[3] = { 16, 16, 1 };

    clEnqueueNDRangeKernel(command_queue[dev_num], kernel_conv_relu[dev_num], 3, NULL, global, local, 0, NULL, NULL);
}

static void fc(cl_mem in, cl_mem out, cl_mem weight, cl_mem bias, int K, int C, int dev_num) {
    clSetKernelArg(kernel_fc[dev_num], 0, sizeof(cl_mem), (void*) &in);
    clSetKernelArg(kernel_fc[dev_num], 1, sizeof(cl_mem), (void*) &out);
    clSetKernelArg(kernel_fc[dev_num], 2, sizeof(cl_mem), (void*) &weight);
    clSetKernelArg(kernel_fc[dev_num], 3, sizeof(cl_mem), (void*) &bias);
    clSetKernelArg(kernel_fc[dev_num], 4, sizeof(cl_int), (void*) &C);

    size_t global[1] = { K };
    size_t local[1] = { 16 };

    clEnqueueNDRangeKernel(command_queue[dev_num], kernel_fc[dev_num], 1, NULL, global, local, 0, NULL, NULL);
}

static void sigmoid(float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = 1 / (1 + expf(-inout[chw]));
    }
}

static void fuse(cl_mem ml, cl_mem gf, cl_mem out, int dev_num) {
    clSetKernelArg(kernel_fuse1[dev_num], 0, sizeof(cl_mem), (void*) &ml);
    clSetKernelArg(kernel_fuse1[dev_num], 1, sizeof(cl_mem), (void*) &out);

    clSetKernelArg(kernel_fuse2[dev_num], 0, sizeof(cl_mem), (void*) &gf);
    clSetKernelArg(kernel_fuse2[dev_num], 1, sizeof(cl_mem), (void*) &out);

    size_t global[3] = { 28, 28, 256 };
    size_t local[3] = { 16, 16, 1 };

    clEnqueueNDRangeKernel(command_queue[dev_num], kernel_fuse1[dev_num], 3, NULL, global, local, 0, NULL, NULL);
    clEnqueueNDRangeKernel(command_queue[dev_num], kernel_fuse2[dev_num], 3, NULL, global, local, 0, NULL, NULL);
}

static void upsample(cl_mem in, cl_mem out, int H, int W, int C, int dev_num) {
    clSetKernelArg(kernel_upsample[dev_num], 0, sizeof(cl_mem), (void*) &in);
    clSetKernelArg(kernel_upsample[dev_num], 1, sizeof(cl_mem), (void*) &out);
    clSetKernelArg(kernel_upsample[dev_num], 2, sizeof(cl_int), (void*) &H);
    clSetKernelArg(kernel_upsample[dev_num], 3, sizeof(cl_int), (void*) &W);

    size_t global[3] = { W, H, C };
    size_t local[3] = { 16, 16, 1 };

    clEnqueueNDRangeKernel(command_queue[dev_num], kernel_upsample[dev_num], 1, NULL, global, local, 0, NULL, NULL);
}

void colorizer_init() {
	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, (unsigned int*) &ndev);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL);
	context = clCreateContext(0, ndev, device, NULL, NULL, NULL);

	for (int i = 0; i < ndev; i++)
		command_queue[i] = clCreateCommandQueue(context, device[i], 0, NULL);

	FILE *fp;
	fp = fopen("kernel.cl", "r");

	kernel_src = (char *) malloc(MAX_SOURCE_SIZE);
	kernel_src_len = fread(kernel_src, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, NULL);
	clBuildProgram(program, ndev, device, NULL, NULL, NULL);

	for (int i = 0; i < ndev; i++) {
        kernel_conv[i] = clCreateKernel(program, "kernel_conv", NULL);
        kernel_conv_relu[i] = clCreateKernel(program, "kernel_conv_relu", NULL);
        kernel_fc[i] = clCreateKernel(program, "kernel_fc", NULL);
        kernel_fuse1[i] = clCreateKernel(program, "kernel_fuse1", NULL);
        kernel_fuse2[i] = clCreateKernel(program, "kernel_fuse2", NULL);
        kernel_upsample[i] = clCreateKernel(program, "kernel_upsample", NULL);
    }
}

void colorizer(int nimg, float *network, float *inputs, float *outputs) {
    cl_mem ll_conv1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 1 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 25088 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_w = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 32 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, NULL);

    cl_mem ll_conv1_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 1 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv1_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 25088 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 32 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, NULL);

    cl_mem ll_conv1_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 1 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv1_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 25088 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 32 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, NULL);

    cl_mem ll_conv1_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 1 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv1_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv2_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv3_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv4_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv5_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ll_conv6_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv1_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem ml_conv2_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv1_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv2_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv3_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem gf_conv4_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * 25088 * sizeof(float), NULL, NULL);
    cl_mem gf_fc1_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fc2_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fc3_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv1_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 256 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv2_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 128 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv3_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv4_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 64 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv5_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 32 * 3 * 3 * sizeof(float), NULL, NULL);
    cl_mem co_conv6_b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(command_queue[0], ll_conv1_w, CL_FALSE, 0, 64 * 1 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv1_w1, CL_FALSE, 0, 64 * 1 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv1_w2, CL_FALSE, 0, 64 * 1 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv1_w3, CL_FALSE, 0, 64 * 1 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 64 * 1 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv1_b, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv1_b1, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv1_b2, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv1_b3, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    network += 64;
    clEnqueueWriteBuffer(command_queue[0], ll_conv2_w, CL_FALSE, 0, 128 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv2_w1, CL_FALSE, 0, 128 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv2_w2, CL_FALSE, 0, 128 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv2_w3, CL_FALSE, 0, 128 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 128 * 64 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv2_b, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv2_b1, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv2_b2, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv2_b3, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    network += 128;
    clEnqueueWriteBuffer(command_queue[0], ll_conv3_w, CL_FALSE, 0, 128 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv3_w1, CL_FALSE, 0, 128 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv3_w2, CL_FALSE, 0, 128 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv3_w3, CL_FALSE, 0, 128 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 128 * 128 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv3_b, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv3_b1, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv3_b2, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv3_b3, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    network += 128;
    clEnqueueWriteBuffer(command_queue[0], ll_conv4_w, CL_FALSE, 0, 256 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv4_w1, CL_FALSE, 0, 256 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv4_w2, CL_FALSE, 0, 256 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv4_w3, CL_FALSE, 0, 256 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 256 * 128 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv4_b, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv4_b1, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv4_b2, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv4_b3, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    network += 256;
    clEnqueueWriteBuffer(command_queue[0], ll_conv5_w, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv5_w1, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv5_w2, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv5_w3, CL_FALSE, 0, 256 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 256 * 256 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv5_b, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv5_b1, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv5_b2, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv5_b3, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    network += 256;
    clEnqueueWriteBuffer(command_queue[0], ll_conv6_w, CL_FALSE, 0, 512 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv6_w1, CL_FALSE, 0, 512 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv6_w2, CL_FALSE, 0, 512 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv6_w3, CL_FALSE, 0, 512 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 256 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ll_conv6_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ll_conv6_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ll_conv6_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ll_conv6_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], ml_conv1_w, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ml_conv1_w1, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ml_conv1_w2, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ml_conv1_w3, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ml_conv1_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ml_conv1_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ml_conv1_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ml_conv1_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], ml_conv2_w, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ml_conv2_w1, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ml_conv2_w2, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ml_conv2_w3, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 256 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], ml_conv2_b, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], ml_conv2_b1, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], ml_conv2_b2, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], ml_conv2_b3, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    network += 256;
    clEnqueueWriteBuffer(command_queue[0], gf_conv1_w, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv1_w1, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv1_w2, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv1_w3, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], gf_conv1_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv1_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv1_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv1_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], gf_conv2_w, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv2_w1, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv2_w2, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv2_w3, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], gf_conv2_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv2_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv2_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv2_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], gf_conv3_w, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv3_w1, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv3_w2, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv3_w3, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], gf_conv3_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv3_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv3_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv3_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], gf_conv4_w, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv4_w1, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv4_w2, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv4_w3, CL_FALSE, 0, 512 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], gf_conv4_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_conv4_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_conv4_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_conv4_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], gf_fc1_w, CL_FALSE, 0, 1024 * 25088 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc1_w1, CL_FALSE, 0, 1024 * 25088 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc1_w2, CL_FALSE, 0, 1024 * 25088 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc1_w3, CL_FALSE, 0, 1024 * 25088 * sizeof(float), network, 0, NULL, NULL);
    network += 1024 * 25088;
    clEnqueueWriteBuffer(command_queue[0], gf_fc1_b, CL_FALSE, 0, 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc1_b1, CL_FALSE, 0, 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc1_b2, CL_FALSE, 0, 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc1_b3, CL_FALSE, 0, 1024 * sizeof(float), network, 0, NULL, NULL);
    network += 1024;
    clEnqueueWriteBuffer(command_queue[0], gf_fc2_w, CL_FALSE, 0, 512 * 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc2_w1, CL_FALSE, 0, 512 * 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc2_w2, CL_FALSE, 0, 512 * 1024 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc2_w3, CL_FALSE, 0, 512 * 1024 * sizeof(float), network, 0, NULL, NULL);
    network += 512 * 1024;
    clEnqueueWriteBuffer(command_queue[0], gf_fc2_b, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc2_b1, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc2_b2, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc2_b3, CL_FALSE, 0, 512 * sizeof(float), network, 0, NULL, NULL);
    network += 512;
    clEnqueueWriteBuffer(command_queue[0], gf_fc3_w, CL_FALSE, 0, 256 * 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc3_w1, CL_FALSE, 0, 256 * 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc3_w2, CL_FALSE, 0, 256 * 512 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc3_w3, CL_FALSE, 0, 256 * 512 * sizeof(float), network, 0, NULL, NULL);
    network += 256 * 512;
    clEnqueueWriteBuffer(command_queue[0], gf_fc3_b, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], gf_fc3_b1, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], gf_fc3_b2, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], gf_fc3_b3, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    network += 256;
    clEnqueueWriteBuffer(command_queue[0], co_conv1_w, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv1_w1, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv1_w2, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv1_w3, CL_FALSE, 0, 256 * 512 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 256 * 512 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv1_b, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv1_b1, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv1_b2, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv1_b3, CL_FALSE, 0, 256 * sizeof(float), network, 0, NULL, NULL);
    network += 256;
    clEnqueueWriteBuffer(command_queue[0], co_conv2_w, CL_FALSE, 0, 128 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv2_w1, CL_FALSE, 0, 128 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv2_w2, CL_FALSE, 0, 128 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv2_w3, CL_FALSE, 0, 128 * 256 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 128 * 256 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv2_b, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv2_b1, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv2_b2, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv2_b3, CL_FALSE, 0, 128 * sizeof(float), network, 0, NULL, NULL);
    network += 128;
    clEnqueueWriteBuffer(command_queue[0], co_conv3_w, CL_FALSE, 0, 64 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv3_w1, CL_FALSE, 0, 64 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv3_w2, CL_FALSE, 0, 64 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv3_w3, CL_FALSE, 0, 64 * 128 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 64 * 128 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv3_b, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv3_b1, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv3_b2, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv3_b3, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    network += 64;
    clEnqueueWriteBuffer(command_queue[0], co_conv4_w, CL_FALSE, 0, 64 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv4_w1, CL_FALSE, 0, 64 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv4_w2, CL_FALSE, 0, 64 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv4_w3, CL_FALSE, 0, 64 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 64 * 64 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv4_b, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv4_b1, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv4_b2, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv4_b3, CL_FALSE, 0, 64 * sizeof(float), network, 0, NULL, NULL);
    network += 64;
    clEnqueueWriteBuffer(command_queue[0], co_conv5_w, CL_FALSE, 0, 32 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv5_w1, CL_FALSE, 0, 32 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv5_w2, CL_FALSE, 0, 32 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv5_w3, CL_FALSE, 0, 32 * 64 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 32 * 64 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv5_b, CL_FALSE, 0, 32 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv5_b1, CL_FALSE, 0, 32 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv5_b2, CL_FALSE, 0, 32 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv5_b3, CL_FALSE, 0, 32 * sizeof(float), network, 0, NULL, NULL);
    network += 32;
    clEnqueueWriteBuffer(command_queue[0], co_conv6_w, CL_FALSE, 0, 2 * 32 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv6_w1, CL_FALSE, 0, 2 * 32 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv6_w2, CL_FALSE, 0, 2 * 32 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv6_w3, CL_FALSE, 0, 2 * 32 * 3 * 3 * sizeof(float), network, 0, NULL, NULL);
    network += 2 * 32 * 3 * 3;
    clEnqueueWriteBuffer(command_queue[0], co_conv6_b, CL_FALSE, 0, 2 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[1], co_conv6_b1, CL_FALSE, 0, 2 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[2], co_conv6_b2, CL_FALSE, 0, 2 * sizeof(float), network, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue[3], co_conv6_b3, CL_FALSE, 0, 2 * sizeof(float), network, 0, NULL, NULL);
    network += 2;

    cl_mem ll_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm4 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm5 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ll_fm6 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem gf_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm4 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm5 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fm6 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fm7 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ml_gf_fused_fm = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm4 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm5 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm6 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem co_fm7 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 112 * 112 * sizeof(float), NULL, NULL);

    cl_mem ll_fm11 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm21 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm31 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm41 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm51 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ll_fm61 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm11 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm21 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem gf_fm11 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm21 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm31 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm41 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm51 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fm61 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fm71 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ml_gf_fused_fm1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm11 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm21 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm31 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm41 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm51 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm61 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem co_fm71 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 112 * 112 * sizeof(float), NULL, NULL);

    cl_mem ll_fm12 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm22 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm32 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm42 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm52 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ll_fm62 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm12 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm22 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem gf_fm12 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm22 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm32 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm42 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm52 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fm62 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fm72 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ml_gf_fused_fm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm12 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm22 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm32 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm42 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm52 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm62 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem co_fm72 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 112 * 112 * sizeof(float), NULL, NULL);

    cl_mem ll_fm13 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm23 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem ll_fm33 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm43 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem ll_fm53 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ll_fm63 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm13 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem ml_fm23 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem gf_fm13 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm23 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 14 * 14 * sizeof(float), NULL, NULL);
    cl_mem gf_fm33 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm43 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 7 * 7 * sizeof(float), NULL, NULL);
    cl_mem gf_fm53 = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, NULL);
    cl_mem gf_fm63 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
    cl_mem gf_fm73 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * sizeof(float), NULL, NULL);
    cl_mem ml_gf_fused_fm3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm13 = clCreateBuffer(context, CL_MEM_READ_WRITE, 256 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm23 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 28 * 28 * sizeof(float), NULL, NULL);
    cl_mem co_fm33 = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm43 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm53 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 56 * 56 * sizeof(float), NULL, NULL);
    cl_mem co_fm63 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 112 * 112 * sizeof(float), NULL, NULL);
    cl_mem co_fm73 = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * 112 * 112 * sizeof(float), NULL, NULL);

    int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n;
	int mpi_quotient = nimg / size;
    int quotient = mpi_quotient / ndev;
    int offset = rank * mpi_quotient;

    for (n = offset; n < offset + ndev * quotient;) {
        cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
        cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

        cl_mem input1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
        cl_mem output1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

        cl_mem input2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
        cl_mem output2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

        cl_mem input3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
        cl_mem output3 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

        float *float_output = outputs + n * 2 * 112 * 112;
        clEnqueueWriteBuffer(command_queue[0], input, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);
        n++;

        float *float_output1 = outputs + n * 2 * 112 * 112;
        clEnqueueWriteBuffer(command_queue[1], input1, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);
        n++;

        float *float_output2 = outputs + n * 2 * 112 * 112;
        clEnqueueWriteBuffer(command_queue[2], input2, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);
        n++;

        float *float_output3 = outputs + n * 2 * 112 * 112;
        clEnqueueWriteBuffer(command_queue[3], input3, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);
        n++;

        conv_relu(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2, 0);
        conv_relu(input1, ll_fm11, ll_conv1_w1, ll_conv1_b1, 224, 224, 64, 1, 2, 1);
        conv_relu(input2, ll_fm12, ll_conv1_w2, ll_conv1_b2, 224, 224, 64, 1, 2, 2);
        conv_relu(input3, ll_fm13, ll_conv1_w3, ll_conv1_b3, 224, 224, 64, 1, 2, 3);

        conv_relu(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1, 0);
        conv_relu(ll_fm11, ll_fm21, ll_conv2_w1, ll_conv2_b1, 112, 112, 128, 64, 1, 1);
        conv_relu(ll_fm12, ll_fm22, ll_conv2_w2, ll_conv2_b2, 112, 112, 128, 64, 1, 2);
        conv_relu(ll_fm13, ll_fm23, ll_conv2_w3, ll_conv2_b3, 112, 112, 128, 64, 1, 3);

        conv_relu(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2, 0);
        conv_relu(ll_fm21, ll_fm31, ll_conv3_w1, ll_conv3_b1, 112, 112, 128, 128, 2, 1);
        conv_relu(ll_fm22, ll_fm32, ll_conv3_w2, ll_conv3_b2, 112, 112, 128, 128, 2, 2);
        conv_relu(ll_fm23, ll_fm33, ll_conv3_w3, ll_conv3_b3, 112, 112, 128, 128, 2, 3);

        conv_relu(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1, 0);
        conv_relu(ll_fm31, ll_fm41, ll_conv4_w1, ll_conv4_b1, 56, 56, 256, 128, 1, 1);
        conv_relu(ll_fm32, ll_fm42, ll_conv4_w2, ll_conv4_b2, 56, 56, 256, 128, 1, 2);
        conv_relu(ll_fm33, ll_fm43, ll_conv4_w3, ll_conv4_b3, 56, 56, 256, 128, 1, 3);

        conv_relu(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2, 0);
        conv_relu(ll_fm41, ll_fm51, ll_conv5_w1, ll_conv5_b1, 56, 56, 256, 256, 2, 1);
        conv_relu(ll_fm42, ll_fm52, ll_conv5_w2, ll_conv5_b2, 56, 56, 256, 256, 2, 2);
        conv_relu(ll_fm43, ll_fm53, ll_conv5_w3, ll_conv5_b3, 56, 56, 256, 256, 2, 3);

        conv_relu(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1, 0);
        conv_relu(ll_fm51, ll_fm61, ll_conv6_w1, ll_conv6_b1, 28, 28, 512, 256, 1, 1);
        conv_relu(ll_fm52, ll_fm62, ll_conv6_w2, ll_conv6_b2, 28, 28, 512, 256, 1, 2);
        conv_relu(ll_fm53, ll_fm63, ll_conv6_w3, ll_conv6_b3, 28, 28, 512, 256, 1, 3);

        conv_relu(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1, 0);
        conv_relu(ll_fm61, ml_fm11, ml_conv1_w1, ml_conv1_b1, 28, 28, 512, 512, 1, 1);
        conv_relu(ll_fm62, ml_fm12, ml_conv1_w2, ml_conv1_b2, 28, 28, 512, 512, 1, 2);
        conv_relu(ll_fm63, ml_fm13, ml_conv1_w3, ml_conv1_b3, 28, 28, 512, 512, 1, 3);

        conv_relu(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1, 0);
        conv_relu(ml_fm11, ml_fm21, ml_conv2_w1, ml_conv2_b1, 28, 28, 256, 512, 1, 1);
        conv_relu(ml_fm12, ml_fm22, ml_conv2_w2, ml_conv2_b2, 28, 28, 256, 512, 1, 2);
        conv_relu(ml_fm13, ml_fm23, ml_conv2_w3, ml_conv2_b3, 28, 28, 256, 512, 1, 3);

        conv_relu(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2, 0);
        conv_relu(ll_fm61, gf_fm11, gf_conv1_w1, gf_conv1_b1, 28, 28, 512, 512, 2, 1);
        conv_relu(ll_fm62, gf_fm12, gf_conv1_w2, gf_conv1_b2, 28, 28, 512, 512, 2, 2);
        conv_relu(ll_fm63, gf_fm13, gf_conv1_w3, gf_conv1_b3, 28, 28, 512, 512, 2, 3);

        conv_relu(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1, 0);
        conv_relu(gf_fm11, gf_fm21, gf_conv2_w1, gf_conv2_b1, 14, 14, 512, 512, 1, 1);
        conv_relu(gf_fm12, gf_fm22, gf_conv2_w2, gf_conv2_b2, 14, 14, 512, 512, 1, 2);
        conv_relu(gf_fm13, gf_fm23, gf_conv2_w3, gf_conv2_b3, 14, 14, 512, 512, 1, 3);

        conv_relu(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2, 0);
        conv_relu(gf_fm21, gf_fm31, gf_conv3_w1, gf_conv3_b1, 14, 14, 512, 512, 2, 1);
        conv_relu(gf_fm22, gf_fm32, gf_conv3_w2, gf_conv3_b2, 14, 14, 512, 512, 2, 2);
        conv_relu(gf_fm23, gf_fm33, gf_conv3_w3, gf_conv3_b3, 14, 14, 512, 512, 2, 3);

        conv_relu(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1, 0);
        conv_relu(gf_fm31, gf_fm41, gf_conv4_w1, gf_conv4_b1, 7, 7, 512, 512, 1, 1);
        conv_relu(gf_fm32, gf_fm42, gf_conv4_w2, gf_conv4_b2, 7, 7, 512, 512, 1, 2);
        conv_relu(gf_fm33, gf_fm43, gf_conv4_w3, gf_conv4_b3, 7, 7, 512, 512, 1, 3);

        fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088, 0);
        fc(gf_fm41, gf_fm51, gf_fc1_w1, gf_fc1_b1, 1024, 25088, 1);
        fc(gf_fm42, gf_fm52, gf_fc1_w2, gf_fc1_b2, 1024, 25088, 2);
        fc(gf_fm43, gf_fm53, gf_fc1_w3, gf_fc1_b3, 1024, 25088, 3);

        fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024, 0);
        fc(gf_fm51, gf_fm61, gf_fc2_w1, gf_fc2_b1, 512, 1024, 1);
        fc(gf_fm52, gf_fm62, gf_fc2_w2, gf_fc2_b2, 512, 1024, 2);
        fc(gf_fm53, gf_fm63, gf_fc2_w3, gf_fc2_b3, 512, 1024, 3);

        fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512, 0);
        fc(gf_fm61, gf_fm71, gf_fc3_w1, gf_fc3_b1, 256, 512, 1);
        fc(gf_fm62, gf_fm72, gf_fc3_w2, gf_fc3_b2, 256, 512, 2);
        fc(gf_fm63, gf_fm73, gf_fc3_w3, gf_fc3_b3, 256, 512, 3);

        fuse(ml_fm2, gf_fm7, ml_gf_fused_fm, 0);
        fuse(ml_fm21, gf_fm71, ml_gf_fused_fm1, 1);
        fuse(ml_fm22, gf_fm72, ml_gf_fused_fm2, 2);
        fuse(ml_fm23, gf_fm73, ml_gf_fused_fm3, 3);

        conv_relu(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1, 0);
        conv_relu(ml_gf_fused_fm1, co_fm11, co_conv1_w1, co_conv1_b1, 28, 28, 256, 512, 1, 1);
        conv_relu(ml_gf_fused_fm2, co_fm12, co_conv1_w2, co_conv1_b2, 28, 28, 256, 512, 1, 2);
        conv_relu(ml_gf_fused_fm3, co_fm13, co_conv1_w3, co_conv1_b3, 28, 28, 256, 512, 1, 3);

        conv_relu(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1, 0);
        conv_relu(co_fm11, co_fm21, co_conv2_w1, co_conv2_b1, 28, 28, 128, 256, 1, 1);
        conv_relu(co_fm12, co_fm22, co_conv2_w2, co_conv2_b2, 28, 28, 128, 256, 1, 2);
        conv_relu(co_fm13, co_fm23, co_conv2_w3, co_conv2_b3, 28, 28, 128, 256, 1, 3);

        upsample(co_fm2, co_fm3, 28, 28, 128, 0);
        upsample(co_fm21, co_fm31, 28, 28, 128, 1);
        upsample(co_fm22, co_fm32, 28, 28, 128, 2);
        upsample(co_fm23, co_fm33, 28, 28, 128, 3);

        conv_relu(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1, 0);
        conv_relu(co_fm31, co_fm41, co_conv3_w1, co_conv3_b1, 56, 56, 64, 128, 1, 1);
        conv_relu(co_fm32, co_fm42, co_conv3_w2, co_conv3_b2, 56, 56, 64, 128, 1, 2);
        conv_relu(co_fm33, co_fm43, co_conv3_w3, co_conv3_b3, 56, 56, 64, 128, 1, 3);

        conv_relu(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1, 0);
        conv_relu(co_fm41, co_fm51, co_conv4_w1, co_conv4_b1, 56, 56, 64, 64, 1, 1);
        conv_relu(co_fm42, co_fm52, co_conv4_w2, co_conv4_b2, 56, 56, 64, 64, 1, 2);
        conv_relu(co_fm43, co_fm53, co_conv4_w3, co_conv4_b3, 56, 56, 64, 64, 1, 3);

        upsample(co_fm5, co_fm6, 56, 56, 64, 0);
        upsample(co_fm51, co_fm61, 56, 56, 64, 1);
        upsample(co_fm52, co_fm62, 56, 56, 64, 2);
        upsample(co_fm53, co_fm63, 56, 56, 64, 3);

        conv_relu(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1, 0);
        conv_relu(co_fm61, co_fm71, co_conv5_w1, co_conv5_b1, 112, 112, 32, 64, 1, 1);
        conv_relu(co_fm62, co_fm72, co_conv5_w2, co_conv5_b2, 112, 112, 32, 64, 1, 2);
        conv_relu(co_fm63, co_fm73, co_conv5_w3, co_conv5_b3, 112, 112, 32, 64, 1, 3);

        conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1, 0);
        conv(co_fm71, output1, co_conv6_w1, co_conv6_b1, 112, 112, 2, 32, 1, 1);
        conv(co_fm72, output2, co_conv6_w2, co_conv6_b2, 112, 112, 2, 32, 1, 2);
        conv(co_fm73, output3, co_conv6_w3, co_conv6_b3, 112, 112, 2, 32, 1, 3);

        clEnqueueReadBuffer(command_queue[0], output, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output, 0, NULL, NULL);
        clEnqueueReadBuffer(command_queue[1], output1, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output1, 0, NULL, NULL);
        clEnqueueReadBuffer(command_queue[2], output2, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output2, 0, NULL, NULL);
        clEnqueueReadBuffer(command_queue[3], output3, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output3, 0, NULL, NULL);

        sigmoid(float_output, 2 * 112 * 112);
        sigmoid(float_output1, 2 * 112 * 112);
        sigmoid(float_output2, 2 * 112 * 112);
        sigmoid(float_output3, 2 * 112 * 112);
    }

    for (n = offset + ndev * quotient; n < offset + mpi_quotient; n++) {
        cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
        cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

        float *float_output = outputs + n * 2 * 112 * 112;
        clEnqueueWriteBuffer(command_queue[0], input, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);

        conv_relu(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2, 0);
        conv_relu(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1, 0);
        conv_relu(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2, 0);
        conv_relu(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1, 0);
        conv_relu(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2, 0);
        conv_relu(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1, 0);

        conv_relu(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1, 0);
        conv_relu(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1, 0);

        conv_relu(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2, 0);
        conv_relu(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1, 0);
        conv_relu(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2, 0);
        conv_relu(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1, 0);
        fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088, 0);
        fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024, 0);
        fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512, 0);

        fuse(ml_fm2, gf_fm7, ml_gf_fused_fm, 0);

        conv_relu(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1, 0);
        conv_relu(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1, 0);
        upsample(co_fm2, co_fm3, 28, 28, 128, 0);
        conv_relu(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1, 0);
        conv_relu(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1, 0);
        upsample(co_fm5, co_fm6, 56, 56, 64, 0);
        conv_relu(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1, 0);
        conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1, 0);

        clEnqueueReadBuffer(command_queue[0], output, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output, 0, NULL, NULL);
        sigmoid(float_output, 2 * 112 * 112);
    }

    float *output = outputs + offset * 2 * 112 * 112;
	int output_num = mpi_quotient * 2 * 112 * 112;
    MPI_Gather(output, output_num, MPI_FLOAT, output, output_num, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

	if (rank == MASTER) {
        for (n = mpi_quotient * size; n < nimg; n++) {
            cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, 224 * 224 * sizeof(float), NULL, NULL);
            cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * 112 * 112 * sizeof(float), NULL, NULL);

            float *float_output = outputs + n * 2 * 112 * 112;
            clEnqueueWriteBuffer(command_queue[0], input, CL_FALSE, 0, 224 * 224 * sizeof(float), inputs + n * 224 * 224, 0, NULL, NULL);

            conv_relu(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2, 0);
            conv_relu(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1, 0);
            conv_relu(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2, 0);
            conv_relu(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1, 0);
            conv_relu(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2, 0);
            conv_relu(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1, 0);

            conv_relu(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1, 0);
            conv_relu(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1, 0);

            conv_relu(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2, 0);
            conv_relu(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1, 0);
            conv_relu(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2, 0);
            conv_relu(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1, 0);
            fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088, 0);
            fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024, 0);
            fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512, 0);

            fuse(ml_fm2, gf_fm7, ml_gf_fused_fm, 0);

            conv_relu(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1, 0);
            conv_relu(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1, 0);
            upsample(co_fm2, co_fm3, 28, 28, 128, 0);
            conv_relu(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1, 0);
            conv_relu(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1, 0);
            upsample(co_fm5, co_fm6, 56, 56, 64, 0);
            conv_relu(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1, 0);
            conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1, 0);

            clEnqueueReadBuffer(command_queue[0], output, CL_TRUE, 0, 2 * 112 * 112 * sizeof(float), float_output, 0, NULL, NULL);
            sigmoid(float_output, 2 * 112 * 112);
        }
	}
}
