#pragma once
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define IMAGE_MAGIC 2051
#define LABEL_MAGIC 2049
#define NUM_CLASSES 10

typedef struct {
    int count;
    int rows;
    int cols;
    float* images; // Flat array: size [count * rows * cols]
} MnistImageSet;

typedef struct {
    int count;
    uint8_t* labels; // Flat array: size [count]
} MnistLabelSet;

typedef struct {
    int count;
    int rows;
    int cols;
    float* images;         // Flat array [count * (rows * cols)] normalized to [0, 1]
    float* labels_one_hot; // Flat array [count * 10]
    uint8_t* labels_raw;   // Flat array [count]
} MnistDataset;

// Core functions
MnistImageSet parse_mnist_images(const uint8_t* buffer, size_t buffer_size);
MnistLabelSet parse_mnist_labels(const uint8_t* buffer, size_t buffer_size);
void one_hot_encode(uint8_t label, float* out_vec, int num_classes);

// File loading & memory management
bool load_mnist_dataset(const char* images_path, const char* labels_path, MnistDataset* out_dataset);
void free_mnist_dataset(MnistDataset* dataset);

#endif // MNIST_LOADER_H