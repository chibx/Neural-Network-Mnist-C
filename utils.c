#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist_loader.h"
#include "neural_network.h"

const char* TRAINING_IMAGE_PATH = "./train-images.idx3-ubyte";
const char* TEST_IMAGE_PATH = "./t10k-images.idx3-ubyte";
const char* TRAINING_LABEL_PATH = "./train-labels.idx1-ubyte";
const char* TEST_LABEL_PATH = "./t10k-labels.idx1-ubyte";
int shared[1] = {16};
const IntArray SHARED_HLAYERS = (IntArray){sizeof(shared) / sizeof(int), &shared[0]};

double rand_double() { return rand() / (double)(RAND_MAX + 1u); }

void print_and_exit(char* c) {
    fprintf(stderr, "%s\n", c);
    exit(1);
}

void shuffle(IntArray data) {
    int length = data.length;

    for (int i = length - 1; i >= 0; i--) {
        int front = floor(rand_double() * (i + 1));
        int tmp = data.arr[i];
        data.arr[i] = data.arr[front];
        data.arr[front] = tmp;
    }
}

bool nn_has_exploded(NeuralNetwork* nn) {
    for (int i = 0; i < nn->layers.length; i++) {
        NeuronArray layer = nn->layers.arr[i];
        for (int j = 0; j < layer.length; j++) {
            Neuron n = layer.arr[j];
            if (!isfinite(n.bias))
                return true;
            for (int k = 0; k < n.weights.length; k++) {
                if (!isfinite(n.weights.arr[k]))
                    return true;
            }
        }
    }
    return false;
}

NeuralNetwork* mnist_train(MNISTTrainOption option) {
    int max_epoch = option.max_epoch == NULL ? 10 : *option.max_epoch;
    double min_error = option.min_error == NULL ? 0.01 : *option.min_error;
    int batch_size = option.batch_size == NULL ? 100 : *option.batch_size;
    BATCH_TYPE is_batched = option.batched == NULL ? SOME : *option.batched;

    MnistDataset dataset = {};
    bool succeeded = load_mnist_dataset(TRAINING_IMAGE_PATH, TRAINING_LABEL_PATH, &dataset);
    if (!succeeded) {
        print_and_exit("Failed to load dataset");
    }
    int count = dataset.count;
    int* indices = malloc(count * sizeof(int));
    if (indices == NULL) {
        print_and_exit("Failed to allocate memory for shuffling dataset");
    }

    for (int i = 0; i < count; i++) {
        indices[i] = i;
    }
    IntArray dataset_indices = {count, indices};

    ExportParameterReturn parameters = nn_load_parameters(PARAMS_PATH);

    int num_inputs = dataset.rows * dataset.cols;
    int num_outputs = 10;
    NeuralNetworkOptions nn_options = {.activation_dx = relu_dx,
                                       .activation_func = relu_activation,
                                       .h_layer_data = SHARED_HLAYERS,
                                       .inputCount = num_inputs,
                                       .num_outputs = &num_outputs,
                                       .weights = parameters.weights,
                                       .biases = parameters.biases};

    NeuralNetwork* nn = create_neural_network(nn_options);

    float epoch_cost = 0;
    for (int cur_epoch = 1; cur_epoch <= max_epoch; cur_epoch++) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

        nn_new_epoch(nn, is_batched, batch_size);
        shuffle(dataset_indices);

        float cost = 0;

        for (int idx = 0; idx < count; idx++) {
            int shuffled_index = dataset_indices.arr[idx];
            FloatArray inputs = {num_inputs, &dataset.images[shuffled_index]};
            cost += nn_train(nn, inputs, &dataset.labels_one_hot[shuffled_index]);
            // if (nn_has_exploded(nn)) {
            //     fprintf(stderr, "exploded at epoch %d, sample %d, loss=%f\n", nn->epoch, nn->training_index, cost);
            //     break;
            // }
        }

        float average_cost = cost / (float)count;
        if (average_cost <= min_error) {
            printf("Exiting training, model has reached minimum error with %f\n", average_cost);
            return nn;
        }
        epoch_cost += average_cost;
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        double time_diff = (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;
        printf("Time taken for epoch %d: %lf seconds\n", cur_epoch, time_diff);
    }

    // struct timespec begin, end;
    // clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    nn_complete_fitting(nn);
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // completetraining_time += (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;

    printf("Finished all epochs with average epoch error of %lf\n", epoch_cost / (float)max_epoch);
    free_mnist_dataset(&dataset);

    // printf("Time taken for all forward propagation: %lf seconds\n", forwardprop_time);
    // printf("Time taken for all complete training: %lf seconds\n", completetraining_time);
    // printf("Time taken for all backward propagation: %lf seconds\n", backprop_time);
    return nn;
}

MNISTTestResult mnist_test(MNISTTestOption option) {
    MnistDataset dataset = {};
    bool succeeded = load_mnist_dataset(TEST_IMAGE_PATH, TEST_LABEL_PATH, &dataset);
    if (!succeeded) {
        print_and_exit("Failed to load dataset");
    }
    int count = dataset.count;

    ExportParameterReturn parameters = nn_load_parameters(PARAMS_PATH);

    int num_inputs = dataset.rows * dataset.cols;
    int num_outputs = 10;
    int total_cases = count;
    int total_success = 0;
    double* outputs = calloc(count * num_outputs, sizeof(double));
    if (outputs == NULL) {
        print_and_exit("Failed to allocated memory for the prediction outputs");
    }

    NeuralNetworkOptions nn_options = {.activation_dx = relu_dx,
                                       .activation_func = relu_activation,
                                       .h_layer_data = SHARED_HLAYERS,
                                       .inputCount = num_inputs,
                                       .num_outputs = &num_outputs,
                                       .weights = parameters.weights,
                                       .biases = parameters.biases};

    NeuralNetwork* nn = create_neural_network(nn_options);

    for (int idx = 0; idx < total_cases; idx++) {
        int output_offset = idx * num_outputs;
        FloatArray inputs = {num_inputs, &dataset.images[idx]};
        nn_predict(nn, inputs, outputs + output_offset, num_outputs);
        double max = -INFINITY;
        int answer_idx = -1;
        for (int i = 0; i < num_outputs; i++) {
            if (outputs[idx * num_outputs + i] > max) {
                max = outputs[idx * num_outputs + i];
                answer_idx = i;
            }
        }

        if (answer_idx == dataset.labels_raw[idx]) {
            total_success++;
        }
    }

    free_mnist_dataset(&dataset);
    free_neural_network(nn);

    return (MNISTTestResult){.passed = total_success, .total = total_cases};
}