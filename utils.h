#include "mnist_loader.h"
#include "neural_network.h"
#include "stdbool.h"

#define PARAMS_PATH "./params-c.json"

typedef struct {
    BATCH_TYPE* batched;
    int* batch_size;
    int* max_epoch;
    double* min_error;
} MNISTTrainOption;

typedef struct {
    // int* max_test;
} MNISTTestOption;

typedef struct {
    int passed;
    int total;
} MNISTTestResult;

double rand_double();
void print_and_exit(char*);
void print_mnist_image(float* images, int grid_size);
NeuralNetwork* mnist_train(MNISTTrainOption option);
MNISTTestResult mnist_test(MNISTTestOption option);
