#pragma once
#include "my_matrix.h"
#include <stdbool.h>

// it's structure to contain neural network
typedef struct NeuralNetwork_s {
	int layers; // the count of neuron layers (with input and output layer)
	int* neuron_counts; // the array with number of neurons on each layer
	bool with_bias; // the flag indicating status: with/without bias
	bool biases_on; // the flag indicating status: on/off biases
	Matrix** bias; // the array with matrixes that contain bias values
	Matrix** weights; // the array with matrixes that contain weights
	Matrix** neurons; // the array with matrixes that contaan tempurary values in neurons e.g. T = <W, X> + B
	bool with_softmax; // the flag indicating status: with/without softmax function
	function_set hfunc; // the function set with activation function for hidden neurons
	function_set ofunc; // the function set with activation function for output layers
} NeuralNetwork;

// the function that initialize network with random values for weights in interval [aw, bw] and biases in interval [ab, bb]
int create_network(int layers, int* neurons, int hidden_func_type, int output_func_type, bool with_softmax, double aw, double bw, bool with_bias, double ab, double bb, NeuralNetwork** nn);
// the function to reset network values
int reset_network(NeuralNetwork* nn, double aw, double bw, double ab, double bb);
// the function to add biases in network
int add_biases(NeuralNetwork* nn, double a, double b);
// dlete biases from network
void delete_biases(NeuralNetwork* nn);
// the function that print network in a *.txt file
void print_network(NeuralNetwork* nn, char* name);
// the function that saves neural network in *.bin file at address "name"
void save_network(NeuralNetwork* network, char* name);
// the function that load neural network from *.bin file at address "name"
int load_network(char* name, NeuralNetwork** nn);
// the function that free data in "network"
void free_network(NeuralNetwork** network);
