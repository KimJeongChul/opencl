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
  // TODO
}
