__kernel void conv(__global float *in,
    __global float *out,
    __global float *weight,
    __global float *bias,
    int H, int W, int K, int C, int stride)
{
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

__kernel void fc(__global float *in,
    __global float *out,
    __global float *weight,
    __global float *bias,
    int K, int C)
{
    for (int k = 0; k < K; ++k) {
        float s = 0;
        for (int c = 0; c < C; ++c) {
            s += in[c] * weight[k * C + c];
        }
        s += bias[k];
        out[k] = s;
    }
}

__kernel void relu(__global float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = fmax(inout[chw], 0);
    }
}

__kernel void sigmoid(__global float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = 1 / (1 + exp(-inout[chw]));
    }
}

__kernel void fuse(__global float* ml,
    __global float* gf,
    __global float* out)
{
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

__kernel void unsample(__global float* in,
    __global float* out, 
    int H, int W, int C)
{
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