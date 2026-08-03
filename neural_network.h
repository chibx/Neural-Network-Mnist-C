#pragma once
#include "stdbool.h"

#define MIN_BATCH_SIZE 2
#define DEFAULT_BATCH_SIZE 500
#define LEARNING_RATE 0.01

typedef double (*ActivationFunction)(double);
typedef enum { IMMEDIATE = 0, ALL = 1, SOME = 2 } BATCH_TYPE;

typedef struct {
    int length;
    float* arr;
} FloatArray;

typedef struct {
    int length;
    double* arr;
} DoubleArray;

typedef struct {
    int length;
    int* arr;
} IntArray;

typedef struct {
    double value;
    double bias;
    DoubleArray weights;
} Neuron;

typedef struct {
    int length;
    Neuron* arr;
} NeuronArray;

typedef struct {
    int length;
    NeuronArray* arr;
} NNLayerArray;

typedef struct {
    int inputCount;
    /** Don't free this pointer */
    DoubleArray* biases;
    /** Don't free this pointer */
    DoubleArray* weights;
    IntArray h_layer_data;
    // bool batched;
    // /** batchSize must be a number greater than 10 */
    // int* batch_size;
    int* num_outputs;
    double* decay;
    ActivationFunction activation_func;
    ActivationFunction activation_dx;
} NeuralNetworkOptions;

typedef struct {
    int num_of_weights;
    int num_of_neurons;
    int neurons_with_bias;
    int max_layer_size;
    int training_index;
    int epoch;
    int is_batched;
    BATCH_TYPE batch_size;
    double decay;
    ActivationFunction activation_func;
    ActivationFunction activation_dx;
    /** The layers holds details about how many inputs are in each layer to allow accurate navigation */
    NNLayerArray layers;
    DoubleArray weight_gradient_accumulator;
    DoubleArray bias_gradient_accumulator;
} NeuralNetwork;

typedef struct {
    DoubleArray* biases;
    DoubleArray* weights;
} ExportParameterReturn;

extern double backprop_time;
extern double forwardprop_time;
extern double completetraining_time;

double random_gaussian();
double relu_activation(double x);
double relu_dx(double x);
double sigmoid_activation(double x);
double sigmoid_dx(double x);
// int total_weight_count(IntArray* layer_data);
// int total_network_neurons(IntArray* layer_data);
// DoubleArray get_neuron_weight(DoubleArray* weights, IntArray* layer_data, int neuron_idx, int cur_layer);
DoubleArray* random_weights(IntArray* layer_data);
NeuralNetwork* create_neural_network(NeuralNetworkOptions option);
void free_neural_network(NeuralNetwork* nn);
void nn_new_epoch(NeuralNetwork* nn, BATCH_TYPE is_batched, int batch_size);
double nn_train(NeuralNetwork* nn, FloatArray inputs, float* expected_outputs);
void nn_predict(NeuralNetwork* nn, FloatArray inputs, double* outputs, int output_size);
void nn_complete_fitting(NeuralNetwork* nn);
bool nn_export_parameters(char* path, NeuralNetwork* nn);
ExportParameterReturn nn_load_parameters(char* path);