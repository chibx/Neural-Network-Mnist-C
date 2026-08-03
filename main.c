#include "neural_network.h"
#include "utils.h"
#include <bits/time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int main(int arg_len, char* argv[]) {
    srand(time(NULL));
    bool TEST = false;
    if (arg_len > 1 && strcmp(argv[1], "test") == 0) {
        TEST = true;
    }

    // int shared$1[] = {3};
    // int shared$1_len = sizeof(shared$1) / sizeof(int);
    // const IntArray SHARED$1_HLAYERS = (IntArray){shared$1_len, &shared$1[0]};
    // int num_inputs = 5;
    // int num_outputs = 2;
    // NeuralNetworkOptions nn_options = {
    //     .activation_dx = relu_dx,
    //     .activation_func = relu_activation,
    //     .h_layer_data = SHARED$1_HLAYERS,
    //     .inputCount = num_inputs,
    //     .num_outputs = &num_outputs,
    // };
    // NeuralNetwork* nn = create_neural_network(nn_options);
    // printf("Weights: %d | Biases: %d\n", nn->num_of_weights, nn->neurons_with_bias);
    // float a[5] = {0.5, 0.23, 0.412, 0.54, 0.3};
    // float expected_outputs[2] = {0.1, 0.2};
    // nn_new_epoch(nn, IMMEDIATE, 100);
    // float cost = nn_train(nn, (FloatArray){5, a}, expected_outputs);
    // printf("Cost: %f\n", cost);
    // bool succeeded = nn_export_parameters(PARAMS_PATH, nn);
    // if (!succeeded) {
    //     fprintf(stderr, "Error exporting parameters.");
    //     return 0;
    // }

    if (!TEST) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
        int max_epoch = 100;
        BATCH_TYPE batched = SOME;
        int batch_size = 10;
        MNISTTrainOption option = {.max_epoch = &max_epoch, .batched = &batched, .batch_size = &batch_size};
        NeuralNetwork* nn = mnist_train(option);

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        double time_diff = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;
        printf("Time taken for training: %lf seconds\n", time_diff);

        bool succeeded = nn_export_parameters(PARAMS_PATH, nn);
        if (!succeeded) {
            fprintf(stderr, "Error exporting parameters.");
            return 0;
        }
        free_neural_network(nn);
    }

    if (TEST) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

        MNISTTestResult result = mnist_test((MNISTTestOption){});

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        double time_diff = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;
        printf("Time taken for testing %d cases: %lf seconds\n", result.total, time_diff);
        printf("Passed %d out of %d test cases\nSuccess Rate: %.3f\n", result.passed, result.total, result.passed / (float)result.total);
    }
}