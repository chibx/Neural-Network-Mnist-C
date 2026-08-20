#include "neural_network.h"
#include "./pkg/cjson/cJSON.h"
#include "utils.h"
#include <errno.h>
#include <linux/limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double backprop_time = 0;
double forwardprop_time = 0;
double completetraining_time = 0;

double random_gaussian() {
    double u1 = rand_double();
    double u2 = rand_double();

    while (u1 == 0)
        u1 = rand_double();

    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

double relu_activation(double x) { return x > 0 ? x : 0; }

double relu_dx(double x) { return x > 0 ? 1 : 0; }

double sigmoid_activation(double x) { return 1.0 / (1.0 + expf(-x)); }

double sigmoid_dx(double x) { return x * (1.0 - x); }

int total_weight_count(IntArray* layer_data) {
    int length = layer_data->length;
    int sum = 0;
    int* arr = layer_data->arr;

    if (arr == NULL) {
        print_and_exit("Memory block passed to `total_weight_count` is null");
    }

    for (int i = 0; i < length - 1; i++) {
        sum += arr[i] * arr[i + 1];
    }

    return sum;
}

extern inline int total_network_neurons(IntArray* layer_data) {
    int length = layer_data->length;
    int sum = 0;
    int* arr = layer_data->arr;

    if (arr == NULL) {
        print_and_exit("Layer Data's internal array is null before using");
    }

    for (int i = 0; i < length; i++) {
        sum += arr[i];
    }

    return sum;
}

extern inline double cost(const double* restrict output, const double* restrict expected, int number_of_outputs) {
    double sum = 0;
    for (int i = 0; i < number_of_outputs; i++) {
        sum += pow((output[i] - expected[i]), 2);
    }

    return sum;
}

extern inline DoubleArray get_neuron_weight(DoubleArray* weights, IntArray* layer_data, int neuron_idx, int cur_layer) {
    int weight_count = weights->length;
    int offset = 0;
    int* layer_data_arr = layer_data->arr;
    int layer_count = layer_data->length;
    double* weight_arr = weights->arr;

    if (layer_data_arr == NULL) {
        print_and_exit("Layer Data's internal array is null before using");
    }
    if (weight_arr == NULL) {
        print_and_exit("Weight generated or passed in with null internal array");
    }

    for (int i = 0; i < cur_layer; i++) {
        offset += layer_data_arr[i] * layer_data_arr[i + 1];
    }

    int next_layer_count = layer_data_arr[cur_layer + 1];

    // Reuse memory
    int index = offset + neuron_idx * next_layer_count;

    return (DoubleArray){next_layer_count, &weight_arr[index]};
}

DoubleArray* random_weights(IntArray* layer_data) {
    int total_weights = total_weight_count(layer_data);
    int* arr = layer_data->arr;
    if (arr == NULL) {
        print_and_exit("Layer Data's internal array is null before using");
    }

    double* weight_arr = malloc(total_weights * sizeof(double));
    if (weight_arr == NULL) {
        print_and_exit("Failed to allocate memory to generate random weights");
    }

    int counter = 0;
    int cur_layer_count = arr[counter];
    int next_max = cur_layer_count * arr[counter + 1];
    for (int i = 0; i < total_weights; i++) {
        weight_arr[i] = random_gaussian() * sqrt(2.0f / cur_layer_count);

        if (i >= next_max - 1) {
            counter++;
            cur_layer_count = arr[counter];
            if (counter + 1 < layer_data->length) {
                next_max += cur_layer_count * arr[counter + 1];
            }
        }
    }

    DoubleArray* f = malloc(sizeof(DoubleArray));
    if (f == NULL) {
        print_and_exit("Failed to allocate memory for FloatArray in random weight gen");
    }
    f->length = total_weights;
    f->arr = weight_arr;
    return f;
}

extern inline double get_neuron_bias(IntArray* layer_data, DoubleArray* biases, int cur_layer, int neuron_idx) {
    // No explicit checks
    int offset = 0;
    int* layer_data_arr = layer_data->arr;
    double* bias_arr = biases->arr;
    if (cur_layer < 1) {
        print_and_exit("Invalid layer id to get bias");
    }

    for (int i = 1; i < cur_layer; i++) {
        offset += layer_data_arr[i];
    }

    return bias_arr[offset + neuron_idx];
}

NNLayerArray make_layers(IntArray layer_data, DoubleArray* biases, DoubleArray* weights) {
    NeuronArray* neuron_arr = malloc(layer_data.length * sizeof(NeuronArray));
    if (neuron_arr == NULL) {
        print_and_exit("Failed to allocate memory for layer");
    }

    int* layer_data_arr = layer_data.arr;
    for (int layer_idx = 0; layer_idx < layer_data.length; layer_idx++) {
        NeuronArray* cur_layer = &neuron_arr[layer_idx];
        int neuron_count = layer_data_arr[layer_idx];
        Neuron* neurons = malloc(neuron_count * sizeof(Neuron));
        if (neurons == NULL) {
            print_and_exit("Failed to allocate memory for neurons in make_layers");
        }

        cur_layer->length = neuron_count;
        cur_layer->arr = neurons;
        int neuron_idx = 0;
        for (int j = 0; j < neuron_count; j++) {
            bool has_next_layer = layer_idx < layer_data.length - 1;
            double bias = 0.0;
            if (layer_idx > 0) {
                bias = get_neuron_bias(&layer_data, biases, layer_idx, neuron_idx);
            }
            DoubleArray n_weights =
                layer_idx <= (layer_data.length - 2) ? get_neuron_weight(weights, &layer_data, neuron_idx, layer_idx) : (DoubleArray){0, NULL};
            neurons[j].weights = n_weights;
            neurons[j].bias = bias;
            neurons[j].value = 0;
            neuron_idx++;
        }
    }

    return (NNLayerArray){layer_data.length, neuron_arr};
}

void reset_gradient_accumulators(DoubleArray* weight_acc, DoubleArray* biases_acc) {
    if (weight_acc->arr == NULL) {
        double* arr = malloc(weight_acc->length * sizeof(double));
        if (arr == NULL) {
            print_and_exit("Failed to allocate memory for weight gradient accumulator");
        }
        weight_acc->arr = arr;
    }
    if (biases_acc->arr == NULL) {
        double* arr = malloc(biases_acc->length * sizeof(double));
        if (arr == NULL) {
            print_and_exit("Failed to allocate memory for bias gradient accumulator");
        }
        biases_acc->arr = arr;
    }

    for (int i = 0; i < weight_acc->length; i++) {
        weight_acc->arr[i] = 0;
    }

    for (int i = 0; i < biases_acc->length; i++) {
        biases_acc->arr[i] = 0;
    }
}

/** References to the weight and biases should not be freed by the user
 *
 *  Expect manually allocated weights and biases. `h_layer_data` memory is up to the user
 */
NeuralNetwork* create_neural_network(NeuralNetworkOptions option) {
    int input_count = option.inputCount;
    if (input_count <= 0) {
        print_and_exit(("Invalid number of inputs passed to config"));
    }
    int num_hidden_layers = option.h_layer_data.length;
    int num_outputs = (option.num_outputs != NULL) ? (*option.num_outputs < 1 ? 1 : *option.num_outputs) : 1;

    int layer_count = num_hidden_layers + 2;
    int* layer_data_arr = malloc(layer_count * sizeof(int));

    if (layer_data_arr == NULL) {
        print_and_exit("Failed to allocate memory");
    }

    layer_data_arr[0] = input_count;

    for (int i = 0; i < layer_count - 2; i++) {
        layer_data_arr[i + 1] = option.h_layer_data.arr[i];
    }

    layer_data_arr[layer_count - 1] = num_outputs;
    IntArray layer_data = {layer_count, layer_data_arr};

    int total_neurons = total_network_neurons(&layer_data);
    int neurons_with_bias = total_neurons - input_count;
    int expected_weight_count = total_weight_count(&layer_data);

    DoubleArray* biases = option.biases;
    DoubleArray* weights = option.weights;
    if (biases == NULL) {
        double* arr = malloc(neurons_with_bias * sizeof(double));
        DoubleArray* bias_arr = malloc(sizeof(DoubleArray));
        if (arr == NULL || bias_arr == NULL) {
            print_and_exit("Failed to allocate memory for zero-filled biases");
        }

        for (int i = 0; i < neurons_with_bias; i++) {
            arr[i] = 0.0;
        }
        bias_arr->length = neurons_with_bias;
        bias_arr->arr = arr;
        biases = bias_arr;
    }

    if (weights == NULL) {
        weights = random_weights(&layer_data);
    }

    if (biases->length < neurons_with_bias) {
        print_and_exit("Invalid number of biases passed to options");
    }
    if (weights->length < expected_weight_count) {
        print_and_exit("Invalid number of weights passed to options");
    }
    NNLayerArray layers = make_layers(layer_data, biases, weights);

    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        print_and_exit("Failed to allocate memory for neural network instance");
    }

    // 1. Calculate maximum layer size to allocate stack memory once
    int max_layer_size = 0;
    for (int i = 0; i < layer_count; i++) {
        if (layer_data_arr[i] > max_layer_size) {
            max_layer_size = layer_data_arr[i];
        }
    }
    nn->max_layer_size = max_layer_size;
    nn->is_batched = ALL;
    nn->batch_size = DEFAULT_BATCH_SIZE;
    nn->activation_func = sigmoid_activation;
    nn->activation_dx = sigmoid_dx;
    nn->num_of_neurons = total_neurons;
    nn->num_of_weights = expected_weight_count;
    nn->neurons_with_bias = neurons_with_bias;
    nn->decay = option.decay == NULL ? 0.95 : *option.decay;
    nn->layers = layers;
    nn->bias_gradient_accumulator = (DoubleArray){neurons_with_bias};
    nn->weight_gradient_accumulator = (DoubleArray){expected_weight_count};
    nn->training_index = 0;
    nn->epoch = 0;

    if (option.activation_dx != NULL && option.activation_func != NULL) {
        nn->activation_dx = option.activation_dx;
        nn->activation_func = option.activation_func;
    }

    free(layer_data.arr);
    layer_data.arr = NULL;
    // free(weights->arr);
    free(weights);
    free(biases->arr);
    free(biases);
    return nn;
};

void free_neural_network(NeuralNetwork* nn) {
    int layer_len = nn->layers.length;
    NeuronArray* layer_arr = nn->layers.arr;
    double* shared_weights_ptr = layer_arr[0].arr[0].weights.arr;

    for (int i = 0; i < layer_len; i++) {
        free(layer_arr[i].arr);
    }
    free(layer_arr);
    free(shared_weights_ptr);
    if (nn->bias_gradient_accumulator.arr != NULL) {
        free(nn->bias_gradient_accumulator.arr);
    }
    if (nn->weight_gradient_accumulator.arr != NULL) {
        free(nn->weight_gradient_accumulator.arr);
    }
    free(nn);
    nn = NULL;
    layer_arr = NULL;
    shared_weights_ptr = NULL;
}

void nn_new_epoch(NeuralNetwork* nn, BATCH_TYPE is_batched, int batch_size) {
    nn->epoch++;
    nn->is_batched = is_batched;
    if (is_batched == SOME) {
        nn->batch_size = batch_size >= MIN_BATCH_SIZE ? batch_size : MIN_BATCH_SIZE;
    }

    nn->training_index = 0;
    reset_gradient_accumulators(&nn->weight_gradient_accumulator, &nn->bias_gradient_accumulator);
}

// Original function 1.)
// void backpropagate(NeuralNetwork* nn, float* expected_outputs) {
//     NeuronArray* layer_data_arr = nn->layers.arr;
//     int num_layers = nn->layers.length;
//     double* current_delta = malloc(layer_data_arr[num_layers - 1].length * sizeof(double));
//     if (current_delta == NULL) {
//         print_and_exit("Failed to allocate memory, current_delta");
//     }
//     int last_layer_length = nn->layers.arr[num_layers - 1].length;
//     for (int i = 0; i < last_layer_length; i++) {
//         current_delta[i] = 2 * (layer_data_arr[num_layers - 1].arr[i].value - expected_outputs[i]);
//     }

//     for (int i = num_layers - 1; i >= 1; i--) {
//         const NeuronArray cur_layer = layer_data_arr[i];
//         const NeuronArray prev_layer = layer_data_arr[i - 1];
//         // int layer_offset = prev_layer.length;
//         int bias_offset = cur_layer.length;
//         double* prev_delta = calloc(prev_layer.length, sizeof(double));
//         if (prev_delta == NULL) {
//             print_and_exit("Failed to allocate memory, prev_delta");
//         }

//         for (int j = 0; j < prev_layer.length; j++) {
//             const DoubleArray prev_layer_nweights = prev_layer.arr[j].weights;
//             // const int dist = wg_gradient_len - wgt_neuron_offset - layer_offset;

//             for (int k = 0; k < cur_layer.length; k++) {
//                 prev_delta[j] += prev_layer_nweights.arr[k] * nn->activation_dx(cur_layer.arr[k].value) * current_delta[k];
//             }

//             for (int k = 0; k < prev_layer_nweights.length; k++) {
//                 const double weight_dx = prev_layer.arr[j].value * nn->activation_dx(cur_layer.arr[k].value) * current_delta[k];
//                 // i - 1 = prev_layer; j = neuron_idx on the prev_layer
//                 // k = weight of that neuron
//                 add_weight_grad(nn, i - 1, j, k, weight_dx);
//             }
//         }

//         for (int b = 0; b < cur_layer.length; b++) {
//             double bias_grad = nn->activation_dx(cur_layer.arr[b].value) * current_delta[b];
//             // i = cur_layer index of which the neuron with the bias exists
//             add_bias_grad(nn, i, b, bias_grad);
//         }

//         // wgt_neuron_offset += prev_layer.length;
//         // bias_neuron_offset += cur_layer.length;
//         free(current_delta);
//         current_delta = prev_delta;
//     }

//     free(current_delta);
// }

void backpropagate(NeuralNetwork* nn, double* expected_outputs) {
    NeuronArray* layer_data_arr = nn->layers.arr;
    int num_layers = nn->layers.length;
    double* weight_grad_arr = nn->weight_gradient_accumulator.arr;
    double* bias_grad_arr = nn->bias_gradient_accumulator.arr;

    // 1. Calculate maximum layer size to allocate stack memory once
    int max_layer_size = nn->max_layer_size;

    int last_layer_length = layer_data_arr[num_layers - 1].length;

    double* current_delta = malloc(max_layer_size * sizeof(double));
    double* prev_delta = malloc(max_layer_size * sizeof(double));
    if (current_delta == NULL || prev_delta == NULL) {
        print_and_exit("Failed to allocate memory for either the layers' nodes delta");
    }

    // 3. Pre-calculate total offsets for flat array access
    int wgt_offset = 0;
    int bias_offset = 0;
    for (int i = 0; i < num_layers - 1; i++) {
        wgt_offset += layer_data_arr[i].length * layer_data_arr[i + 1].length;
        bias_offset += layer_data_arr[i + 1].length;
    }

    // 4. Initialize the output layer delta
    for (int i = 0; i < last_layer_length; i++) {
        current_delta[i] = 2.0 * (layer_data_arr[num_layers - 1].arr[i].value - expected_outputs[i]);
    }

    // 5. The optimized backward pass
    for (int i = num_layers - 1; i >= 1; i--) {
        const NeuronArray cur_layer = layer_data_arr[i];
        const NeuronArray prev_layer = layer_data_arr[i - 1];

        // Shift offsets backwards for direct flat array access
        wgt_offset -= (prev_layer.length * cur_layer.length);
        bias_offset -= cur_layer.length;

        // Zero out the previous delta buffer (fixes the uninitialized memory bug!)
        for (int j = 0; j < prev_layer.length; j++) {
            prev_delta[j] = 0.0;
        }

        // Pre-calculate node deltas to destroy the function pointer bottleneck
        double* node_delta = malloc(cur_layer.length * sizeof(double));
        if (node_delta == NULL) {
            print_and_exit("Failed to allocate memory for node_delta");
        }
        for (int j = 0; j < cur_layer.length; j++) {
            node_delta[j] = nn->activation_dx(cur_layer.arr[j].value) * current_delta[j];

            // Accumulate bias gradients immediately
            bias_grad_arr[bias_offset + j] += node_delta[j];
        }

        // The core matrix multiplication loops
        for (int j = 0; j < prev_layer.length; j++) {
            const DoubleArray prev_layer_nweights = prev_layer.arr[j].weights;
            const double prev_val = prev_layer.arr[j].value;
            const int weight_base_idx = wgt_offset + (j * cur_layer.length);

            for (int k = 0; k < cur_layer.length; k++) {
                // Accumulate error for the previous layer
                prev_delta[j] += prev_layer_nweights.arr[k] * node_delta[k];

                // Accumulate weight gradient
                weight_grad_arr[weight_base_idx + k] += prev_val * node_delta[k];
            }
        }

        free(node_delta);
        // Swap pointers for the next layer iteration
        double* temp = current_delta;
        current_delta = prev_delta;
        prev_delta = temp;
    }

    free(current_delta);
    free(prev_delta);
}

double nn_train(NeuralNetwork* nn, FloatArray inputs, double* restrict expected_outputs) {
    // struct timespec begin, end;
    // clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    NeuronArray first_layer = nn->layers.arr[0];
    if (first_layer.length != inputs.length) {
        print_and_exit("Input passed to nn.train is not of the required length");
    }

    NeuronArray last_layer = nn->layers.arr[nn->layers.length - 1];
    double* outputs = malloc(last_layer.length * sizeof(double));
    if (outputs == NULL) {
        print_and_exit("Failed to allocate memory for training output");
    }

    for (int i = 0; i < inputs.length; i++) {
        first_layer.arr[i].value = inputs.arr[i];
    }

    const int num_layers = nn->layers.length;
    // Forward Propagation
    for (int i = 0; i < num_layers - 1; i++) {
        const NeuronArray cur_layer = nn->layers.arr[i];
        const NeuronArray next_layer = nn->layers.arr[i + 1];
        const int cur_layer_len = cur_layer.length;
        const int next_layer_len = next_layer.length;

        for (int j = 0; j < next_layer_len; j++) {
            double sum = 0.0;
            for (int k = 0; k < cur_layer_len; k++) {
                const Neuron* neuron = &cur_layer.arr[k];
                sum += neuron->value * neuron->weights.arr[j];
            }

            next_layer.arr[j].value = nn->activation_func(sum + next_layer.arr[j].bias);
        }
    }
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // forwardprop_time += (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;
    for (int i = 0; i < last_layer.length; i++) {
        outputs[i] = last_layer.arr[i].value;
    }

    double _cost = cost(outputs, expected_outputs, last_layer.length);

    // Backpropagation

    // clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
    backpropagate(nn, expected_outputs);
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // backprop_time += (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;

    nn->training_index++;

    if (nn->is_batched == IMMEDIATE || (nn->is_batched == SOME && nn->training_index >= nn->batch_size)) {
        // clock_gettime(CLOCK_MONOTONIC_RAW, &begin);
        nn_complete_fitting(nn);
        // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        // completetraining_time += (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + end.tv_sec - begin.tv_sec;
    }

    free(outputs);
    return _cost;
}

void nn_complete_fitting(NeuralNetwork* nn) {
    if (nn->training_index == 0) {
        return;
    }

    const double sample_count = nn->training_index;
    int neuron_idx = 0;
    int layer_len = nn->layers.length;
    NeuronArray* layer_arr = nn->layers.arr;

    for (int i = 1; i < layer_len; i++) {
        const NeuronArray cur_layer = layer_arr[i];
        for (int j = 0; j < cur_layer.length; j++) {
            double avg_bias_grad = nn->bias_gradient_accumulator.arr[neuron_idx] / sample_count;
            cur_layer.arr[j].bias -= LEARNING_RATE * pow(nn->decay, nn->epoch) * avg_bias_grad;
            neuron_idx++;
        }
    }

    // neuron_idx = 0;
    int offset = 0;
    for (int layer_idx = 0; layer_idx < layer_len - 1; layer_idx++) {
        const NeuronArray cur_layer = layer_arr[layer_idx];
        const NeuronArray next_layer = layer_arr[layer_idx + 1];
        for (int n_idx = 0; n_idx < cur_layer.length; n_idx++) {
            Neuron* cur_neuron = &cur_layer.arr[n_idx];
            for (int weight_idx = 0; weight_idx < next_layer.length; weight_idx++) {
                double avg_weight_grad = nn->weight_gradient_accumulator.arr[offset + n_idx * next_layer.length + weight_idx] / sample_count;
                double tmp = cur_neuron->weights.arr[weight_idx];
                cur_neuron->weights.arr[weight_idx] = tmp - LEARNING_RATE * pow(nn->decay, nn->epoch) * avg_weight_grad;
            }
            // neuron_idx++;
        }
        // neuron_idx = 0;
        offset += cur_layer.length * next_layer.length;
    }

    nn->training_index = 0;
    reset_gradient_accumulators(&nn->weight_gradient_accumulator, &nn->bias_gradient_accumulator);
}

void nn_predict(NeuralNetwork* nn, FloatArray inputs, double* outputs, int output_size) {
    NeuronArray first_layer = nn->layers.arr[0];
    if (first_layer.length != inputs.length) {
        print_and_exit("Input passed to nn_predict is not the required length");
    }

    NeuronArray last_layer = nn->layers.arr[nn->layers.length - 1];
    if (last_layer.length != output_size) {
        print_and_exit("Last layer and Output array are of different length");
    }
    for (int i = 0; i < inputs.length; i++) {
        first_layer.arr[i].value = inputs.arr[i];
    }

    int num_layers = nn->layers.length;
    for (int i = 0; i < num_layers - 1; i++) {
        NeuronArray cur_layer = nn->layers.arr[i];
        NeuronArray next_layer = nn->layers.arr[i + 1];

        for (int j = 0; j < next_layer.length; j++) {
            double sum = 0;
            for (int k = 0; k < cur_layer.length; k++) {
                Neuron* neuron = &cur_layer.arr[k];
                sum += neuron->value * neuron->weights.arr[j];
            }

            next_layer.arr[j].value = nn->activation_func(sum + next_layer.arr[j].bias);
        }
    }

    for (int i = 0; i < last_layer.length; i++) {
        outputs[i] = last_layer.arr[i].value;
    }
}

/* extern inline bool manual_json_writer(NeuralNetwork* nn, FILE* file) {
    fwrite("{\"biases\":[", sizeof(char), 11, file);

    const int batch_size = 5 * 1024;
    char* text_batch = calloc(batch_size, sizeof(char));
    int batch_index = 0;
    if (text_batch == NULL) {
        return false;
    }

    char tmp[20] = {0};
    const int layer_len = nn->layers.length;
    for (int i = 1; i < layer_len; i++) {
        NeuronArray cur_layer = nn->layers.arr[i];
        for (int j = 0; j < cur_layer.length; j++) {

            int written = snprintf(tmp, sizeof(tmp), "%lf", cur_layer.arr[j].bias);
            if (written >= (int)sizeof(tmp)) {
                written = sizeof(tmp) - 1;
            }

            if (batch_index + written + 2 >= batch_size) {
                if (text_batch[batch_index - 1] == ',') {
                    text_batch[batch_index - 1] = '\0';
                }
                fwrite(text_batch, sizeof(char), strlen(text_batch), file);
                // Clear the batch sequence and reset the batch counter
                // for (int i = 0; i < batch_index; i++) {
                //     text_batch[i] = '\0';
                // }
                text_batch[0] = ',';
                text_batch[1] = '\0';
                batch_index = 1;
            }

            for (int k = 0; k < written; k++) {
                text_batch[batch_index + k] = tmp[k];
                tmp[k] = '\0';
            }
            text_batch[batch_index + written] = ',';
            batch_index += written + 1;
            text_batch[batch_index] = '\0';

            // if (j == cur_layer.length - 1 && i == layer_len - 1) {
            //     if (text_batch[batch_index - 1] == ',') {
            //         text_batch[batch_index - 1] = '\0';
            //     }
            // }
        }
    }
    if (text_batch[batch_index - 1] == ',') {
        text_batch[batch_index - 1] = '\0';
    }
    fprintf(file, "%s],\"weights\":[", text_batch);
    // for (int i = 0; i < batch_index; i++) {
    //     text_batch[i] = '\0';
    // }
    text_batch[0] = '\0';
    batch_index = 0;

    // Weights parse logic
    for (int i = 0; i < layer_len - 1; i++) {
        NeuronArray cur_layer = nn->layers.arr[i];
        for (int j = 0; j < cur_layer.length; j++) {
            Neuron cur_neuron = cur_layer.arr[j];
            const int neuron_weight_count = cur_neuron.weights.length;
            for (int k = 0; k < neuron_weight_count; k++) {
                int written = snprintf(tmp, sizeof(tmp), "%lf", cur_neuron.weights.arr[k]);
                if (written >= (int)sizeof(tmp)) {
                    written = sizeof(tmp) - 1;
                }

                if (batch_index + written + 2 >= batch_size) {
                    if (text_batch[batch_index - 1] == ',') {
                        text_batch[batch_index - 1] = '\0';
                    }
                    fwrite(text_batch, sizeof(char), strlen(text_batch), file);
                    // Clear the batch sequence and reset the batch counter
                    text_batch[0] = ',';
                    text_batch[1] = '\0';
                    batch_index = 1;
                }

                for (int l = 0; l < written; l++) {
                    text_batch[batch_index + l] = tmp[l];
                    tmp[l] = '\0';
                }

                text_batch[batch_index + written] = ',';
                batch_index += written + 1;
                text_batch[batch_index] = '\0';
            }
        }
    }
    if (text_batch[batch_index - 1] == ',') {
        text_batch[batch_index - 1] = '\0';
    }
    fprintf(file, "%s]}", text_batch);

    return true;
} */

bool nn_export_parameters(char* path, NeuralNetwork* nn) {
    FILE* file = fopen(path, "w");
    if (file == NULL) {
        return false;
    }

    // bool success = manual_json_writer(nn, file);
    bool success = true;

    cJSON* json = cJSON_CreateObject();
    if (json == NULL) {
        success = false;
        goto json_err;
    }

    cJSON* json_bias_arr = cJSON_CreateArray();
    cJSON* json_weight_arr = cJSON_CreateArray();
    if (json_bias_arr == NULL) {
        success = false;
        goto json_err;
    }
    cJSON_AddItemToObject(json, "biases", json_bias_arr);
    cJSON_AddItemToObject(json, "weights", json_weight_arr);

    const int layer_len = nn->layers.length;
    for (int i = 1; i < layer_len; i++) {
        NeuronArray cur_layer = nn->layers.arr[i];
        for (int j = 0; j < cur_layer.length; j++) {
            cJSON* bias = cJSON_CreateNumber(cur_layer.arr[j].bias);
            if (bias == NULL) {
                success = false;
                goto json_err;
            }

            cJSON_AddItemToArray(json_bias_arr, bias);
        }
    }

    for (int i = 0; i < layer_len - 1; i++) {
        NeuronArray cur_layer = nn->layers.arr[i];
        for (int j = 0; j < cur_layer.length; j++) {
            Neuron cur_neuron = cur_layer.arr[j];
            const int neuron_weight_count = cur_neuron.weights.length;
            for (int k = 0; k < neuron_weight_count; k++) {
                cJSON* weight = cJSON_CreateNumber(cur_neuron.weights.arr[k]);
                if (weight == NULL) {
                    success = false;
                    goto json_err;
                }

                cJSON_AddItemToArray(json_weight_arr, weight);
            }
        }
    }

    char* json_string = cJSON_PrintUnformatted(json);
    fwrite(json_string, sizeof(char), strlen(json_string), file);
    free(json_string);
    // fclose(file);

json_err: {
    cJSON_Delete(json);
    fclose(file);
}

    return success;
}

ExportParameterReturn nn_load_parameters(char* path) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        if (errno == ENOENT) {
            return (ExportParameterReturn){};
        }
        print_and_exit("Parameter file could not be loaded");
    }

    char* param_error = "Something went wrong while loading parameters from file";
    fseek(file, 0, SEEK_END);
    int sz = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (sz <= 0) {
        fclose(file);
        print_and_exit(param_error);
    }

    char* buf = malloc(sz);
    if (!buf) {
        fclose(file);
        print_and_exit(param_error);
    }

    if (fread(buf, 1, sz, file) != sz) {
        free(buf);
        fclose(file);
        print_and_exit(param_error);
    }

    fclose(file);

    cJSON* json = cJSON_Parse(buf);
    free(buf);
    if (json == NULL) {
        cJSON_Delete(json);
        print_and_exit(param_error);
    }
    // free(buf);

    double* biases_arr = NULL;
    double* weight_arr = NULL;
    const cJSON* biases = cJSON_GetObjectItemCaseSensitive(json, "biases");
    int biases_len = 0;
    if (cJSON_IsArray(biases)) {
        biases_len = cJSON_GetArraySize(biases);
        biases_arr = malloc(biases_len * sizeof(double));
        if (biases_arr == NULL) {
            cJSON_Delete(json);
            print_and_exit("Couldn't allocate memory for weights in nn_load_parameters");
        }

        int index = 0;
        cJSON* bias = NULL;
        cJSON_ArrayForEach(bias, biases) {
            if (cJSON_IsNumber(bias)) {
                biases_arr[index] = bias->valuedouble;
            } else {
                cJSON_Delete(json);
                print_and_exit("Expected to see numeric character in biases array of the parameter file");
            }
            index++;
        }
    }

    const cJSON* weights = cJSON_GetObjectItemCaseSensitive(json, "weights");
    int weight_len = 0;
    if (cJSON_IsArray(weights)) {
        weight_len = cJSON_GetArraySize(weights);
        weight_arr = malloc(weight_len * sizeof(double));
        if (weight_arr == NULL) {
            cJSON_Delete(json);
            print_and_exit("Couldn't allocate memory for weights in nn_load_parameters");
        }

        int index = 0;
        cJSON* weight = NULL;
        cJSON_ArrayForEach(weight, weights) {
            if (cJSON_IsNumber(weight)) {
                weight_arr[index] = weight->valuedouble;
            } else {
                cJSON_Delete(json);
                print_and_exit("Expected to see numeric character in weights array of the parameter file");
            }
            index++;
        }
    }

    if (weight_arr == NULL || biases_arr == NULL) {
        cJSON_Delete(json);
        print_and_exit(param_error);
    }

    cJSON_Delete(json);

    ExportParameterReturn parsed = {};
    DoubleArray* b = malloc(sizeof(DoubleArray));
    DoubleArray* w = malloc(sizeof(DoubleArray));
    if (b == NULL || w == NULL) {
        print_and_exit(param_error);
    }

    parsed.biases = b;
    parsed.weights = w;
    parsed.biases->length = biases_len;
    parsed.biases->arr = biases_arr;
    parsed.weights->length = weight_len;
    parsed.weights->arr = weight_arr;

    return parsed;
}