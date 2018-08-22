#include "colorizer.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Convolution Layer
 * in : (C, H, W)
 * out : (K, H / stride, W / stride)
 * weight : (K, C, 3, 3)
 * bias : (K)
 */
static void conv(float *in, float *out, float *weight, float *bias, int H, int W, int K, int C, int stride) {
    int HOUT = H / stride, WOUT = W / stride;
    for (int k = 0; k < K; ++k) {
        for (int hout = 0; hout < HOUT; ++hout) {
            for (int wout = 0; wout < WOUT; ++wout) {
                float sum = bias[k];
                for (int c = 0; c < C; ++c) {
                    for (int r = 0; r < 3; ++r) {
                        for (int s = 0; s < 3; ++s) {
                            // calculate position in input image
                            int h = hout * stride + r - 1;
                            int w = wout * stride + s - 1;
                            if (h < 0 || h >= H || w < 0 || w >= W) {
                                // out of bound, do nothing
                            } else {
                                sum += in[c * H * W + h * W + w] * weight[k * C * 3 * 3 + c * 3 * 3 + r * 3 + s];
                            }
                        }
                    }
                }
                out[k * HOUT * WOUT + hout * WOUT + wout] = sum;
            }
        }
    }
}

/*
 * Fully-Connected Layer (matrix-vector multiplication)
 * in : (C)
 * out : (K)
 * weight : (K, C)
 * bias : (K)
 */
static void fc(float *in, float *out, float *weight, float *bias, int K, int C) {
    for (int k = 0; k < K; ++k) {
        float s = 0;
        for (int c = 0; c < C; ++c) {
            s += in[c] * weight[k * C + c];
        }
        s += bias[k];
        out[k] = s;
    }
}

/*
 * ReLU (in-place)
 * inout : (C, H, W)
 */
static void relu(float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = fmaxf(inout[chw], 0);
    }
}

/*
 * Sigmoid (in-place)
 * inout : (C, H, W)
 */
static void sigmoid(float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = 1 / (1 + expf(-inout[chw]));
    }
}

/*
 * ml : (256, 28, 28)
 * gf : (256)
 * out : (512, 28, 28)
 */
static void fuse(float *ml, float *gf, float *out) {
    for (int k = 0; k < 256; ++k) {
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                out[k * 28 * 28 + h * 28 + w] = ml[k * 28 * 28 + h * 28 + w];
            }
        }
    }
    for (int k = 256; k < 512; ++k) {
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                out[k * 28 * 28 + h * 28 + w] = gf[k - 256];
            }
        }
    }
}

/*
 * in : (C, H, W)
 * out : (C, H * 2, W * 2)
 */
static void upsample(float *in, float *out, int H, int W, int C) {
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float t = in[c * H * W + h * W + w];
                out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 0)] = t;
                out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 1)] = t;
                out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 0)] = t;
                out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 1)] = t;
            }
        }
    }
}

void colorizer_init() {
}

void colorizer(int nimg, float *network, float *inputs, float *outputs) {
    // ll = Low-Level Feature Network
    // ml = Mid-Level Feature Network
    // gf = Global Feature Network
    // co = Colorization Network
    // w = weight, b = bias

    // split network into each layer's parameter
    float *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
    float *ll_conv1_b = network; network += 64;
    float *ll_conv2_w = network; network += 128 * 64 * 3 * 3;
    float *ll_conv2_b = network; network += 128;
    float *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
    float *ll_conv3_b = network; network += 128;
    float *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
    float *ll_conv4_b = network; network += 256;
    float *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
    float *ll_conv5_b = network; network += 256;
    float *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
    float *ll_conv6_b = network; network += 512;
    float *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
    float *ml_conv1_b = network; network += 512;
    float *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
    float *ml_conv2_b = network; network += 256;
    float *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv1_b = network; network += 512;
    float *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv2_b = network; network += 512;
    float *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv3_b = network; network += 512;
    float *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv4_b = network; network += 512;
    float *gf_fc1_w = network; network += 1024 * 25088;
    float *gf_fc1_b = network; network += 1024;
    float *gf_fc2_w = network; network += 512 * 1024;
    float *gf_fc2_b = network; network += 512;
    float *gf_fc3_w = network; network += 256 * 512;
    float *gf_fc3_b = network; network += 256;
    float *co_conv1_w = network; network += 256 * 512 * 3 * 3;
    float *co_conv1_b = network; network += 256;
    float *co_conv2_w = network; network += 128 * 256 * 3 * 3;
    float *co_conv2_b = network; network += 128;
    float *co_conv3_w = network; network += 64 * 128 * 3 * 3;
    float *co_conv3_b = network; network += 64;
    float *co_conv4_w = network; network += 64 * 64 * 3 * 3;
    float *co_conv4_b = network; network += 64;
    float *co_conv5_w = network; network += 32 * 64 * 3 * 3;
    float *co_conv5_b = network; network += 32;
    float *co_conv6_w = network; network += 2 * 32 * 3 * 3;
    float *co_conv6_b = network; network += 2;

    // intermediate buffer for feature maps
    float *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    float *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
    float *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    float *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    float *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *ll_fm6 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *ml_fm2 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    float *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    float *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    float *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    float *gf_fm5 = (float*)malloc(1024 * sizeof(float));
    float *gf_fm6 = (float*)malloc(512 * sizeof(float));
    float *gf_fm7 = (float*)malloc(256 * sizeof(float));
    float *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
    float *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    float *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    float *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    float *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    float *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));

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
