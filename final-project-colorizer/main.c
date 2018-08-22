#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "colorizer.h"

const float datamean = 0.44505388568813;

float clamp(float x, float l, float u) {
    if (x < l) return l;
    if (x > u) return u;
    return x;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s input.bin output.bin\n", argv[0]);
        exit(0);
    }
    printf("Reading inputs... "); fflush(stdout);
    // input read
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        printf("%s does not exist.\n", argv[1]);
        exit(0);
    }
    fseek(fin, 0, SEEK_END);
    long fsz = ftell(fin);
    if (fsz % (224 * 224 * 3) != 0) {
        printf("Input file is corrupted.\n");
        exit(0);
    }
    int nimg = fsz / (224 * 224 * 3);
    rewind(fin);
    unsigned char *imgs = (unsigned char*)malloc(nimg * 224 * 224 * 3);
    fread(imgs, 1, nimg * 224 * 224 * 3, fin);
    fclose(fin);

    // rgb to y
    float *inputs = (float*)malloc(nimg * 224 * 224 * sizeof(float));
    unsigned char *pimgs = imgs;
    float *pinputs = inputs;
    for (int n = 0; n < nimg; ++n) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                pinputs[0] = pimgs[0] / 255.0 * 0.299 + pimgs[1] / 255.0 * 0.587 + pimgs[2] / 255.0 * 0.114 - datamean;
                pimgs += 3;
                pinputs += 1;
            }
        }
    }

    // network read
    FILE *fnet = fopen("network.bin", "rb");
    if (!fnet) {
        printf("Network file does not exist.\n");
        exit(0);
    }
    fseek(fnet, 0, SEEK_END);
    fsz = ftell(fnet);
    if (fsz != 172873608) {
        printf("Network file is corrupted.\n");
        exit(0);
    }
    rewind(fnet);
    float *network = (float*)malloc(172873608);
    fread(network, 1, 172873608, fnet);
    fclose(fnet);
    float *outputs = (float*)malloc(nimg * 2 * 112 * 112 * sizeof(float));
    printf(" done!\n");
    printf("%d images are detected.\n", nimg);

    // initialize; does not count into elapsed time
    printf("Initializing... "); fflush(stdout);
    colorizer_init();
    printf("done!\n");

    // main calculation
    printf("Calculating... "); fflush(stdout);
    timer_start(0);
    colorizer(nimg, network, inputs, outputs);
    double elapsed = timer_stop(0);
    printf("done!\n");
    printf("Elapsed time : %.6f sec\n", elapsed);

    // write result
    printf("Writing results... "); fflush(stdout);
    unsigned char *colored = (unsigned char*)malloc(nimg * 224 * 224 * 3 * sizeof(unsigned char));
    for (int n = 0; n < nimg; ++n) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                float y, rgb[3], xyz[3], yuv[3], l, a, b;

                // y -> yyy (as yuv) -> rgb -> lab -> l
                y = inputs[n * 224 * 224 + h * 224 + w] + datamean;
                rgb[0] = clamp(y + 1.14 * y, 0, 1);
                rgb[1] = clamp(y - 0.395 * y - 0.581 * y, 0, 1);
                rgb[2] = clamp(y + 2.033 * y, 0, 1);
                for (int i = 0; i < 3; ++i) {
                    if (rgb[i] > 0.04045) {
                        rgb[i] = powf((rgb[i] + 0.055) / 1.055, 2.4);
                    } else {
                        rgb[i] /= 12.92;
                    }
                    rgb[i] *= 100.0;
                }
                xyz[1] = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
                xyz[1] /= 100.0;
                if (xyz[1] > 0.008856) {
                    xyz[1] = powf(xyz[1], 1.0 / 3.0);
                } else {
                    xyz[1] = 7.787 * xyz[1] + 16.0 / 116.0;
                }
                l = 116.0 * xyz[1] - 16.0;

                // a, b from output
                a = outputs[n * 2 * 112 * 112 + 0 * 112 * 112 + (h / 2) * 112 + (w / 2)];
                b = outputs[n * 2 * 112 * 112 + 1 * 112 * 112 + (h / 2) * 112 + (w / 2)];
                a = a * 2 - 1;
                b = b * 2 - 1;
                a *= 100;
                b *= 100;

                // lab -> rgb -> yuv -> uv
                xyz[1] = (l + 16) / 116;
                xyz[0] = a / 500 + xyz[1];
                xyz[2] = xyz[1] - b / 200;
                xyz[0] = 0.95047 * ((xyz[0] * xyz[0] * xyz[0] > 0.008856) ? xyz[0] * xyz[0] * xyz[0] : (xyz[0] - 16.0 / 116.0) / 7.787);
                xyz[1] = 1.00000 * ((xyz[1] * xyz[1] * xyz[1] > 0.008856) ? xyz[1] * xyz[1] * xyz[1] : (xyz[1] - 16.0 / 116.0) / 7.787);
                xyz[2] = 1.08883 * ((xyz[2] * xyz[2] * xyz[2] > 0.008856) ? xyz[2] * xyz[2] * xyz[2] : (xyz[2] - 16.0 / 116.0) / 7.787);
                rgb[0] = xyz[0] *  3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986;
                rgb[1] = xyz[0] * -0.9689 + xyz[1] *  1.8758 + xyz[2] *  0.0415;
                rgb[2] = xyz[0] *  0.0557 + xyz[1] * -0.2040 + xyz[2] *  1.0570;
                rgb[0] = (rgb[0] > 0.0031308) ? (1.055 * powf(rgb[0], 1.0 / 2.4) - 0.055) : 12.92 * rgb[0];
                rgb[1] = (rgb[1] > 0.0031308) ? (1.055 * powf(rgb[1], 1.0 / 2.4) - 0.055) : 12.92 * rgb[1];
                rgb[2] = (rgb[2] > 0.0031308) ? (1.055 * powf(rgb[2], 1.0 / 2.4) - 0.055) : 12.92 * rgb[2];
                rgb[0] = clamp(rgb[0], 0, 1);
                rgb[1] = clamp(rgb[1], 0, 1);
                rgb[2] = clamp(rgb[2], 0, 1);
                yuv[0] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                yuv[1] = 0.492 * (rgb[2] - yuv[0]);
                yuv[2] = 0.877 * (rgb[0] - yuv[0]);

                yuv[0] = y;

                rgb[0] = clamp(yuv[0] + 1.14 * yuv[2], 0, 1);
                rgb[1] = clamp(yuv[0] - 0.395 * yuv[1] - 0.581 * yuv[2], 0, 1);
                rgb[2] = clamp(yuv[0] + 2.033 * yuv[1], 0, 1);

                colored[n * 224 * 224 * 3 + h * 224 * 3 + w * 3 + 0] = (unsigned char)(rgb[0] * 255);
                colored[n * 224 * 224 * 3 + h * 224 * 3 + w * 3 + 1] = (unsigned char)(rgb[1] * 255);
                colored[n * 224 * 224 * 3 + h * 224 * 3 + w * 3 + 2] = (unsigned char)(rgb[2] * 255);
            }
        }
    }

    FILE *fout = fopen(argv[2], "wb");
    fwrite(colored, 1, nimg * 224 * 224 * 3, fout);
    fclose(fout);
    printf("done!\n");

    return 0;
}
