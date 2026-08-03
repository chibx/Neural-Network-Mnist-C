#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Reads a 32-bit big-endian integer from a byte buffer.
 */
static uint32_t read_uint32_be(const uint8_t* buffer) {
    return ((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16) | ((uint32_t)buffer[2] << 8) | (uint32_t)buffer[3];
}

/**
 * Reads an entire binary file into a dynamically allocated buffer.
 */
static uint8_t* read_file_to_buffer(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz <= 0) {
        fclose(f);
        return NULL;
    }

    uint8_t* buf = malloc(sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, sz, f) != sz) {
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    if (out_size)
        *out_size = sz;
    return buf;
}

MnistImageSet parse_mnist_images(const uint8_t* buffer, size_t buffer_size) {
    MnistImageSet set = {0};

    if (buffer_size < 16)
        return set;

    uint32_t magic = read_uint32_be(buffer);
    if (magic != IMAGE_MAGIC) {
        fprintf(stderr, "Invalid MNIST image file: expected magic %d, got %u\n", IMAGE_MAGIC, magic);
        return set;
    }

    set.count = (int)read_uint32_be(buffer + 4);
    set.rows = (int)read_uint32_be(buffer + 8);
    set.cols = (int)read_uint32_be(buffer + 12);

    int image_size = set.rows * set.cols;
    //   size_t expected_bytes = 16 + (size_t)set.count * image_size;

    //   if (buffer_size < expected_bytes) {
    //     fprintf(stderr, "Truncated image file buffer\n");
    //     return set;
    //   }

    size_t total_pixels = (size_t)set.count * image_size;
    set.images = (float*)malloc(total_pixels * sizeof(float));
    if (!set.images)
        return set;

    const uint8_t* raw_pixels = buffer + 16;
    for (size_t i = 0; i < total_pixels; i++) {
        // Normalize 0-255 -> 0.0 - 1.0
        set.images[i] = (float)raw_pixels[i] / 255.0f;
    }

    return set;
}

MnistLabelSet parse_mnist_labels(const uint8_t* buffer, size_t buffer_size) {
    MnistLabelSet set = {0};

    if (buffer_size < 8)
        return set;

    uint32_t magic = read_uint32_be(buffer);
    if (magic != LABEL_MAGIC) {
        fprintf(stderr, "Invalid MNIST label file: expected magic %d, got %u\n", LABEL_MAGIC, magic);
        return set;
    }

    set.count = (int)read_uint32_be(buffer + 4);
    size_t expected_bytes = 8 + (size_t)set.count;

    if (buffer_size < expected_bytes) {
        fprintf(stderr, "Truncated label file buffer\n");
        return set;
    }

    set.labels = (uint8_t*)malloc(set.count * sizeof(uint8_t));
    if (!set.labels)
        return set;

    memcpy(set.labels, buffer + 8, set.count);

    return set;
}

void one_hot_encode(uint8_t label, float* out_vec, int num_classes) {
    for (int i = 0; i < num_classes; i++) {
        out_vec[i] = (i == label) ? 1.0f : 0.0f;
    }
}

bool load_mnist_dataset(const char* images_path, const char* labels_path, MnistDataset* out_dataset) {
    size_t img_buf_size = 0, lbl_buf_size = 0;

    uint8_t* img_buf = read_file_to_buffer(images_path, &img_buf_size);
    uint8_t* lbl_buf = read_file_to_buffer(labels_path, &lbl_buf_size);

    if (!img_buf || !lbl_buf) {
        free(img_buf);
        free(lbl_buf);
        return false;
    }

    MnistImageSet img_set = parse_mnist_images(img_buf, img_buf_size);
    MnistLabelSet lbl_set = parse_mnist_labels(lbl_buf, lbl_buf_size);

    // Temp buffers are no longer needed after parsing
    free(img_buf);
    free(lbl_buf);

    if (!img_set.images || !lbl_set.labels) {
        free(img_set.images);
        free(lbl_set.labels);
        return false;
    }

    if (img_set.count != lbl_set.count) {
        fprintf(stderr, "Image count (%d) does not match label count (%d)\n", img_set.count, lbl_set.count);
        free(img_set.images);
        free(lbl_set.labels);
        return false;
    }

    out_dataset->count = img_set.count;
    out_dataset->rows = img_set.rows;
    out_dataset->cols = img_set.cols;
    out_dataset->images = img_set.images;
    out_dataset->labels_raw = lbl_set.labels;

    // Allocate contiguous block for all one-hot labels [count * 10]
    out_dataset->labels_one_hot = (float*)calloc(out_dataset->count * NUM_CLASSES, sizeof(float));
    if (!out_dataset->labels_one_hot) {
        free(out_dataset->images);
        free(out_dataset->labels_raw);
        return false;
    }

    // Populate one-hot encodings
    for (int i = 0; i < out_dataset->count; i++) {
        uint8_t digit = out_dataset->labels_raw[i];
        if (digit < NUM_CLASSES) {
            out_dataset->labels_one_hot[i * NUM_CLASSES + digit] = 1.0f;
        }
    }

    return true;
}

void free_mnist_dataset(MnistDataset* dataset) {
    if (dataset) {
        free(dataset->images);
        free(dataset->labels_one_hot);
        free(dataset->labels_raw);
        dataset->images = NULL;
        dataset->labels_one_hot = NULL;
        dataset->labels_raw = NULL;
        dataset->count = 0;
    }
}