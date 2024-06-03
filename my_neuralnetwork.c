#define _CRT_SECURE_NO_WARNINGS
#include "my_neuralnetwork.h"
#include "my_matrix.h"
#include "my_math.h"
#include <stdlib.h>
#include <stdio.h>

// the function that initialize network with random values for weights in interval [aw, bw] and biases in interval [ab, bb]
int create_network(int layers, int* neurons, int hidden_func_type, int output_func_type, bool with_softmax, double aw, double bw, bool with_bias, double ab, double bb, NeuralNetwork** nn) {
	// checking the correctness of the input data
	if (layers < 2) return 1;
	for (int i = 0; i < layers; i++) {
		if (neurons[i] < 1) return 1;
	}
	if (aw > bw) return 1;
	if (with_bias && (ab > bb)) return 1;

	// if network in nn is NULL then free network
	if ((*nn)) {
		free_network(nn);
	}
	(*nn) = (NeuralNetwork*)malloc(sizeof(NeuralNetwork)); // allocate memory for neural network

	// copying flags
	(*nn)->with_bias = with_bias;
	(*nn)->biases_on = with_bias;
	(*nn)->with_softmax = with_softmax;
	(*nn)->layers = layers;
	// getting functions for hidden and output layers
	(*nn)->hfunc = get_func(hidden_func_type);
	(*nn)->ofunc = get_func(output_func_type);
	// copying data of neuron counts
	(*nn)->neuron_counts = (int*)malloc(layers * sizeof(int));
	for (int i = 0; i < layers; i++) {
		(*nn)->neuron_counts[i] = neurons[i];
	}

	// allocate memory for neurons weights and biases
	(*nn)->neurons = init_matrixes(layers - 1, false);
	(*nn)->weights = init_matrixes(layers - 1, false);
	if (with_bias) {
		(*nn)->bias = init_matrixes(layers - 1, false);
	}
	else {
		(*nn)->bias = NULL;
	}
	// initialize data for neurons weights and biases
	for (int i = 0; i < layers - 1; i++) {
		(*nn)->neurons[i] = create_matrix(neurons[i + 1], 1, 0.0, NULL, false, 0.0, 0.0);
		(*nn)->weights[i] = create_matrix(neurons[i + 1], neurons[i], 0.0, NULL, true, aw, bw);
		if (with_bias) {
			(*nn)->bias[i] = create_matrix(neurons[i + 1], 1, 0.0, NULL, true, ab, bb);
		}
	}
	return 0;
}

// function to reset weights and biases
int reset_network(NeuralNetwork* nn, double aw, double bw, double ab, double bb) {
	if (!nn) return 1;
	if (aw > bw) return 1;
	if (nn->with_bias && (ab > bb)) return 1;
	for (int i = 0; i < nn->layers - 1; i++) {
		free_matrix(&nn->weights[i]); // free old memory
		nn->weights[i] = create_matrix(nn->neuron_counts[i + 1], nn->neuron_counts[i], 0.0, NULL, true, aw, bw); // and generate new values
		// similar for biases
		if (nn->with_bias) {
			free_matrix(&nn->bias[i]);
			nn->bias[i] = create_matrix(nn->neuron_counts[i + 1], 1, 0.0, NULL, true, ab, bb);
		}
	}
	return 0;
}

// function to add biases
int add_biases(NeuralNetwork* nn, double a, double b) {
	if (nn->with_bias) return 1;
	if (a > b) return 1;
	// switching on biases
	nn->with_bias = true;
	nn->biases_on = true;
	// memory allocation for new biases
	nn->bias = init_matrixes(nn->layers - 1, false);
	for (int i = 0; i < nn->layers - 1; i++) {
		nn->bias[i] = create_matrix(nn->neuron_counts[i + 1], 1, 0.0, NULL, true, a, b);
	}
	return 0;
}

// function to delete biases
void delete_biases(NeuralNetwork* nn) {
	if (!nn->with_bias) return;
	// turning off biases
	nn->with_bias = false;
	nn->biases_on = false;
	for (int i = 0; i < nn->layers - 1; i++) { // free all biases memory
		free_matrix(&nn->bias[i]);
	}
	free(nn->bias);
}

// function to print network
void print_network(NeuralNetwork* nn, char* name) {
	FILE* out = fopen(name, "w");

	// printing network layers
	fprintf(out, "Count of layers: %d (", nn->layers);
	for (int i = 0; i < nn->layers; i++) {
		if (i == nn->layers - 1) {
			fprintf(out, "%d", nn->neuron_counts[i]);
		}
		else {
			fprintf(out, "%d | ", nn->neuron_counts[i]);
		}
	}
	fprintf(out, ")\n");
	//printing statuses
	fprintf(out, "With biases: %s\n", nn->with_bias ? "true" : "false");
	fprintf(out, "Biases on: %s\n", nn->biases_on ? "true" : "false");
	fprintf(out, "With softmax: %s\n", nn->with_softmax ? "true" : "false");
	// printing information about functions
	fprintf(out, "Hidden neurons function: ");
	switch (nn->hfunc.type) {
	case 0:
		fprintf(out, "sigmoid\n");
		break;
	case 1:
		fprintf(out, "ELU\n");
		break;
	case 2:
		fprintf(out, "SiLU\n");
		break;
	case 3:
		fprintf(out, "tanh\n");
		break;
	case 4:
		fprintf(out, "ReLU\n");
		break;
	default:
		fprintf(out, "linear\n");
		break;
	}
	fprintf(out, "Output neurons function: ");
	switch (nn->ofunc.type) {
	case 0:
		fprintf(out, "sigmoid\n");
		break;
	case 1:
		fprintf(out, "ELU\n");
		break;
	case 2:
		fprintf(out, "SiLU\n");
		break;
	case 3:
		fprintf(out, "tanh\n");
		break;
	case 4:
		fprintf(out, "ReLU\n");
		break;
	default:
		fprintf(out, "linear\n");
		break;
	}

	// printing weights and biases
	for (int i = 0; i < nn->layers - 1; i++) {
		fprintf(out, "\vl%d weights: \n", i);
		for (int j = 0; j < nn->weights[i]->m; j++) {
			for (int k = 0; k < nn->weights[i]->n; k++) {
				fprintf(out, " %11.20lf", nn->weights[i]->arr[j * nn->weights[i]->n + k]);
			}
			fprintf(out, "\n");
		}
		fprintf(out, "\vl%d biases: \n", i);
		if (nn->with_bias) {
			for (int j = 0; j < nn->bias[i]->m; j++) {
				fprintf(out, " %11.20lf", nn->bias[i]->arr[j]);
			}
		}
		fprintf(out, "\n");
	}

	fclose(out);
}

// function to save neural network
void save_network(NeuralNetwork* network, char* name) {
	FILE* data = fopen(name, "wb"); // open file

	// write network parameters
	fwrite(&network->hfunc.type, sizeof(int), 1, data);
	fwrite(&network->ofunc.type, sizeof(int), 1, data);
	fwrite(&network->layers, sizeof(int), 1, data);
	fwrite(network->neuron_counts, sizeof(int), network->layers, data);
	char c = 1 ? network->with_bias : 0;
	fwrite(&c, sizeof(char), 1, data);
	c = 1 ? network->with_softmax : 0;
	fwrite(&c, sizeof(char), 1, data);

	// write weights and biases data
	for (int i = 0; i < network->layers - 1; i++) {
		fwrite(network->weights[i]->arr, sizeof(double), network->neuron_counts[i] * network->neuron_counts[i + 1], data);
		if (network->with_bias) {
			fwrite(network->bias[i]->arr, sizeof(double), network->bias[i]->m, data);
		}
	}

	fclose(data); // close file
}

// function to load neural network
int load_network(char* name, NeuralNetwork** nn) {
	FILE* data = fopen(name, "rb"); // open file
	if (!data) return 1; // if the file could not be opened, return 0

	// if network in nn is NULL then free network
	if ((*nn)) {
		free_network(nn);
	}
	(*nn) = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

	// getting functions
	int key = 0;
	fread(&key, sizeof(int), 1, data);
	(*nn)->hfunc = get_func(key);
	fread(&key, sizeof(int), 1, data);
	(*nn)->ofunc = get_func(key);

	// reading count of layers and count of neurons
	fread(&(*nn)->layers, sizeof(int), 1, data);
	(*nn)->neuron_counts = (int*)malloc((*nn)->layers * sizeof(int));
	fread((*nn)->neuron_counts, sizeof(int), (*nn)->layers, data);

	// allocating memory for neurons
	(*nn)->neurons = init_matrixes((*nn)->layers - 1, false);

	// allocating memory for biases
	char c = 0;
	fread(&c, sizeof(char), 1, data);
	if (c) {
		(*nn)->with_bias = true;
		(*nn)->biases_on = true;
		(*nn)->bias = init_matrixes((*nn)->layers - 1, false);
	}
	else {
		(*nn)->with_bias = false;
		(*nn)->biases_on = false;
	}
	// reading softmax flag
	fread(&c, sizeof(char), 1, data);
	(*nn)->with_softmax = c ? true : false;

	// reading weights and biases data
	(*nn)->weights = init_matrixes((*nn)->layers - 1, false);
	double* temp = NULL;
	for (int i = 0; i < (*nn)->layers - 1; i++) {
		(*nn)->neurons[i] = create_matrix((*nn)->neuron_counts[i + 1], 1, 0.0, NULL, false, 0.0, 0.0);

		temp = init_doubles((*nn)->neuron_counts[i + 1] * (*nn)->neuron_counts[i], false);
		fread(temp, sizeof(double), (*nn)->neuron_counts[i + 1] * (*nn)->neuron_counts[i], data);
		(*nn)->weights[i] = create_matrix((*nn)->neuron_counts[i + 1], (*nn)->neuron_counts[i], 0.0, temp, false, 0.0, 0.0);
		free(temp);

		if (c) {
			temp = init_doubles((*nn)->neuron_counts[i + 1], false);
			fread(temp, sizeof(double), (*nn)->neuron_counts[i + 1], data);
			(*nn)->bias[i] = create_matrix((*nn)->neuron_counts[i + 1], 1, 0.0, temp, false, 0.0, 0.0);
			free(temp);
		}
	}

	fclose(data); // close file
	return 0;
}

// function to clear network
void free_network(NeuralNetwork** network) {
	if (!network) return;
	if (!(*network)) return;
	free((*network)->neuron_counts);
	(*network)->neuron_counts = NULL;
	for (int i = 0; i < (*network)->layers - 1; i++) {
		free_matrix(&(*network)->weights[i]);
		if ((*network)->with_bias) {
			free_matrix(&(*network)->bias[i]);
		}
	}
	free((*network)->weights);
	(*network)->weights = NULL;
	if ((*network)->with_bias) {
		free((*network)->bias);
		(*network)->bias = NULL;
	}
	free((*network));
	(*network) = NULL;
}
