#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "my_math.h"
#include "my_matrix.h"
#include "my_neuralnetwork.h"

// global parameters to train neural network
double LearningRate = 0.001;
int epochs = 1;
int sets = 0;

// the function to predict answer
Matrix* go_forward(NeuralNetwork* nn, Matrix* inp);
// the function and variables for backpropagation
Matrix** deltaBiases = NULL;
Matrix** deltaWeights = NULL;
void backpropagation(NeuralNetwork* nn, Matrix* inp, Matrix* res, Matrix* out);

// main function
int main() {
	// init variables for data input
	double* input = NULL;
	double* output = NULL;
	Matrix** input_set = NULL;
	Matrix** output_set = NULL;
	Matrix* input_temp = NULL;
	bool is_file = false;
	// variable to read network prediction
	Matrix* ans = NULL;
	// accuracy variable
	double acc = 0.0;
	// variables for neural network
	int neuron_counts[100] = { 0 };
	NeuralNetwork* my_nn = NULL;
	// variables to working with file names
	char str_saved[201] = "saved\\";
	char str_output[201] = "output\\";
	char command[101] = "";
	char name[96] = "";
	// start work session
	printf("Start of work session>\vEntry any command\n");
	while (1) { // infinite cycle to work without restarting
		printf("cmd> ");
		scanf("%s", &command); // command input

		// ===> 1 <===
		if (!strcmp(command, "load")) { // command to load network from file
			printf("arg> ");
			scanf("%s", &name);
			strncat(name, ".bin", 5);
			if (load_network(name, &my_nn)) { // error in reading file
				fprintf(stderr, "!!!Error: the path is incorrect or the file does not exist!\n");
			}
			else {
				if (input_temp) { // allocating memory for ones input set
					free_matrix(&input_temp);
				}
				input_temp = create_matrix(my_nn->neuron_counts[0], 1, 0.0, NULL, false, 0.0, 0.0);
				printf("Succesfuly loaded.\n------------------\n");
				printf("Count of layers: %d (", my_nn->layers);
				for (int i = 0; i < my_nn->layers; i++) {
					if (i == my_nn->layers - 1) {
						printf("%d", my_nn->neuron_counts[i]);
					}
					else {
						printf("%d | ", my_nn->neuron_counts[i]);
					}
				}
				printf(")\n");
				//printing statuses
				printf("With biases: %s\n", my_nn->with_bias ? "true" : "false");
				printf("Biases on: %s\n", my_nn->biases_on ? "true" : "false");
				printf("With softmax: %s\n", my_nn->with_softmax ? "true" : "false");
				// printing information about functions
				printf("Hidden neurons function: ");
				switch (my_nn->hfunc.type) {
				case 0:
					printf("sigmoid\n");
					break;
				case 1:
					printf("ELU\n");
					break;
				case 2:
					printf("SiLU\n");
					break;
				case 3:
					printf("tanh\n");
					break;
				case 4:
					printf("ReLU\n");
					break;
				default:
					printf("linear\n");
					break;
				}
				printf("Output neurons function: ");
				switch (my_nn->ofunc.type) {
				case 0:
					printf("sigmoid\n");
					break;
				case 1:
					printf("ELU\n");
					break;
				case 2:
					printf("SiLU\n");
					break;
				case 3:
					printf("tanh\n");
					break;
				case 4:
					printf("ReLU\n");
					break;
				default:
					printf("linear\n");
					break;
				}
			}
			continue;
		}
		// ===> 2 <===
		if (!strcmp(command, "create")) { // command to create new neural network
			printf("arg> ");
			int cnt = 0;
			scanf("%d", &cnt); // reading count of layers
			if (cnt > 100) {
				fprintf(stderr, "!!!Error: too many layers, their number should be less than 101!\n");
				continue;
			}
			for (int i = 0; i < cnt; i++) { // reading neurons counts
				scanf("%d", &neuron_counts[i]);
			}
			int key1 = -1, key2 = -1;
			printf("functions> ");
			scanf("%d %d", &key1, &key2); // reading keys for hidden/output function
			
			bool with_softmax = false;
			double aw = 0.0, bw = 0.0;
			printf("weights> ");
			scanf("%lf %lf", &aw, &bw); // reading interval for random weights generation
			int c = 0;
			bool with_biases = false;
			double ab = 0.0, bb = 0.0;
			printf("with biases> ");
			scanf("%d", &c);
			if (c) { // if with biases then reading interval for random biases generation
				with_biases = true;
				printf("biases> ");
				scanf("%lf %lf", &ab, &bb);
			}
			printf("with softmax> ");
			scanf("%d", &c);
			if (c) {
				with_softmax = true;
			}
			
			create_network(cnt, neuron_counts, key1, key2, with_softmax, aw, bw, with_biases, ab, bb, &my_nn); // creating new neural network
			if (input_temp) { // allocating memory for temporary once input set
				free_matrix(&input_temp);
			}
			input_temp = create_matrix(my_nn->neuron_counts[0], 1, 0.0, NULL, false, 0.0, 0.0);
			continue;
		}
		// ===> 3 <===
		if (!strcmp(command, "finish")) { // finish programe cycle
			break;
		}
		if (!my_nn) { // if no model is open yet, then we DON'T perform the following functions
			fprintf(stderr, "!!!Error: you can't work without a model, download a model or create one!\n");
			continue;
		}
		// ===> 4 <===
		if (!strcmp(command, "save")) { // function to save *.bin file with current model
			printf("arg> ");
			scanf("%s", &name);
			strncat(name, ".bin", 5);
			strncat(str_saved, name, 110);
			save_network(my_nn, str_saved); // saving model
			printf("Succesfuly saved in saved\\%s\n", name);
			str_saved[7] = '\0';
			continue;
		}
		// ===> 5 <===
		if (!strcmp(command, "output")) { // function to save *.txt file with current model
			printf("arg> ");
			scanf("%s", &name);
			strncat(name, ".txt", 5);
			strncat(str_output, name, 110);
			print_network(my_nn, str_output); // printing network in *.txt
			printf("Succesfuly printed in output\\%s\n", name);
			str_output[8] = '\0';
			continue;
		}
		// ===> 6 <===
		if (!strcmp(command, "open_set")) { // function to open and read training or testing set
			printf("arg> ");
			scanf("%s", &name);
			strncat(name, ".txt", 5);
			FILE* in = fopen(name, "r");
			if (!in) { // error at file opening
				fprintf(stderr, "!!!Error: the path is incorrect or the file does not exist!\n");
				continue;
			}
			fscanf(in, "%d", &sets);
			if (input_set != NULL) { // free memory if necessary
				for (int i = 0; i < sets; i++) {
					free_matrix(&input_set[i]);
					free_matrix(&output_set[i]);
				}
				free(input_set);
				free(output_set);
			}
			// allocating memory for input and output matrixes
			input_set = init_matrixes(sets, false);
			output_set = init_matrixes(sets, false);
			// allocatring memory for temporary input/output data
			input = init_doubles(my_nn->neuron_counts[0], false);
			output = init_doubles(my_nn->neuron_counts[my_nn->layers - 1], false);
			// reading data
			for (int i = 0; i < sets; i++) {
				for (int j = 0; j < my_nn->neuron_counts[0]; j++) { // reading input data
					fscanf(in, "%lf", &input[j]);
				}
				for (int j = 0; j < my_nn->neuron_counts[my_nn->layers - 1]; j++) { // reading output data
					fscanf(in, "%lf", &output[j]);
				}
				input_set[i] = create_matrix(my_nn->neuron_counts[0], 1, 0.0, input, false, 0.0, 0.0);
				output_set[i] = create_matrix(my_nn->neuron_counts[my_nn->layers - 1], 1, 0.0, output, false, 0.0, 0.0);
			}
			fclose(in);
			is_file = true;
			continue;
		}
		// ===> 7 <===
		if (!strcmp(command, "forward")) { // predicts one answer with input data
			printf("arg> ");
			for (int i = 0; i < my_nn->neuron_counts[0]; i++) { // reading data
				scanf("%lf", &input_temp->arr[i]);
			}
			ans = go_forward(my_nn, input_temp); // predicting
			printMatrix(ans, true);
			continue;
		}
		// ===> 8 <===
		if (!strcmp(command, "go_set")) { // predicting all answers for training/testing set
			if (!is_file) {
				fprintf(stderr, "!!!Error: you do not have a file with operational data!\n");
				break;
			}
			acc = 0.0;
			for (int i = 0; i < sets; i++) { // printing predictions
				ans = go_forward(my_nn, input_set[i]); // predicting
				printf("------------------\ninput: ");
				printMatrix(input_set[i], true);
				printf("prediction: ");
				printMatrix(ans, true);
				// calculating accuracy
				double t = accuracy(output_set[i], ans);
				acc += (1 - t);
				printf("output: ");
				printMatrix(output_set[i], true);
			}
			acc = (acc / (double)sets) * 100.0;
			printf("------------------\naccuracy: %0.20lf\n", acc);
			continue;
		}
		// ===> 9 <===
		if (!strcmp(command, "set_train_params")) { // sets training parameters (LearningRate, epochs count)
			printf("arg> ");
			scanf("%lf %lu", &LearningRate, &epochs);
			continue;
		}
		// ===> 10 <===
		if (!strcmp(command, "train")) { // trains neural network
			if (!is_file) {
				fprintf(stderr, "!!!Error: you do not have a file with operational data!\n");
				continue;
			}
			int p = 0;
			int max_p = epochs * sets;
			int proc = 0;
			for (int i = 0; i < epochs; i++) {
				for (int j = 0; j < sets; j++) {
					ans = go_forward(my_nn, input_set[j]); // predicting answer
					backpropagation(my_nn, input_set[j], ans, output_set[j]); // using it to backpropagation
					p++;
					proc = p / (max_p / 100);
					printf("\rIn process...%15d/%d | %3d%%", p, max_p, proc); // printing progress
				}
			}
			printf("\n");
			continue;
		}
		// ===> 11 <===
		if (!strcmp(command, "off_biases")) { // turn off biases
			if (my_nn->with_bias) {
				my_nn->biases_on = false;
			}
			else {
				fprintf(stderr, "!!!Error: you don't have biases, you can't change their state!\n");
			}
			continue;
		}
		// ===> 12 <===
		if (!strcmp(command, "on_biases")) { // swith on biases
			if (my_nn->with_bias) {
				my_nn->biases_on = true;
			}
			else {
				fprintf(stderr, "!!!Error: you don't have biases, you can't change their state!\n");
			}
			continue;
		}
		// ===> 13 <===
		if (!strcmp(command, "add_biases")) { // add biases in model
			printf("arg> ");
			double a = 0.0, b = 0.0;
			scanf("%lf %lf", &a, &b); // reading interval

			if (my_nn->with_bias) { // we can't add them if they already exist
				fprintf(stderr, "!!!Error: you already have biases!\n");
				continue;
			}
			if (add_biases(my_nn, a, b)) { // adding biases
				fprintf(stderr, "!!!Error: wrong input data!\n");
			}
			continue;
		}
		// ===> 14 <===
		if (!strcmp(command, "delete_biases")) { // delete biases
			delete_biases(my_nn);
			continue;
		}
		// ===> 15 <===
		if (!strcmp(command, "reset")) { // reset weights and biases with new random values
			printf("arg> ");
			double aw = 0.0, bw = 0.0;
			double ab = 0.0, bb = 0.0;
			scanf("%lf %lf", &aw, &bw); // reading weights interval
			if (my_nn->with_bias) { // reading biases interval
				printf("biases> ");
				scanf("%lf %lf", &ab, &bb);
			}
			if (reset_network(my_nn, aw, bw, ab, bb)) { // resets values
				fprintf(stderr, "!!!Error: wrong input data!\n");
			}
			continue;
		}
		// ===> 16 <===
		if (!strcmp(command, "set_hidden_f")) { // sets new hidden layer activation function
			printf("arg> ");
			int key = -1;
			scanf("%d", &key);
			my_nn->hfunc = get_func(key);
			continue;
		}
		// ===> 17 <===
		if (!strcmp(command, "set_output_f")) { // sets new output layer activation function
			printf("arg> ");
			int key = -1;
			scanf("%d", &key);
			my_nn->ofunc = get_func(key);
			continue;
		}
		// ===> 18 <===
		if (!strcmp(command, "off_softmax")) { // turn off softmax function
			my_nn->with_softmax = false;
			continue;
		}
		// ===> 19 <===
		if (!strcmp(command, "on_softmax")) { // switch on softmax function
			my_nn->with_softmax = true;
			continue;
		}
	}

	// free all memory
	if (deltaBiases != NULL) {
		for (int i = 0; i < my_nn->layers - 1; i++) {
			free_matrix(&deltaBiases[i]);
			free_matrix(&deltaWeights[i]);
		}
		free(deltaBiases);
		free(deltaWeights);
	}
	free_network(&my_nn);
	if (input != NULL) {
		for (int i = 0; i < sets; i++) {
			free_matrix(&input_set[i]);
			free_matrix(&output_set[i]);
		}
		free(input_set);
		free(output_set);
	}
	if (input_temp != NULL) {
		free_matrix(&input_temp);
	}
	if (input != NULL) {
		free(input);
		free(output);
	}

	return 0;
}

// neural network forward pass
Matrix* go_forward(NeuralNetwork* nn, Matrix* inp) {
	Matrix* ans = NULL;
	Matrix* temp = NULL;
	for (int i = 0; i < nn->layers - 1; i++) {
		if (i == 0) {
			dot(nn->weights[i], false, inp, false, NULL, (nn->biases_on ? nn->bias[i] : NULL), NULL, NULL, &nn->neurons[i]); // T[i] = <W[i], inp> + B[i]
		}
		else {
			dot(nn->weights[i], false, nn->neurons[i - 1], false, nn->hfunc.f, (nn->biases_on ? nn->bias[i] : NULL), NULL, NULL, &nn->neurons[i]); // T[i] = <W[i], f(T[i - 1])> + B[i]
		}
	}
	do_func_on_matrix(nn->neurons[nn->layers - 2], nn->ofunc.f, &temp); // applying an activation function to output neurons
	if (nn->with_softmax) {
		softmax(temp, &ans); // applying softmax
	}
	return nn->with_softmax ? ans : temp;
}

// backpropagation to train neural network
// nn - neural network pointer
// inp - input data
// out - output data
// res - neural network output
void backpropagation(NeuralNetwork* nn, Matrix* inp, Matrix* res, Matrix* out) {
	if (!deltaBiases) { // allocating memory for delta matrixes
		deltaBiases = init_matrixes(nn->layers - 1, true);
		deltaWeights = init_matrixes(nn->layers - 1, true);
	}
	Matrix* temp = NULL;
	sub(res, out, NULL, 1.0, &temp); // calculating error
	// calculating delta
	for (int i = nn->layers - 2; i >= 0; i--) {
		if (i == nn->layers - 2) {
			mul(temp, nn->neurons[i], nn->ofunc.df, 1.0, &deltaBiases[i]); // dB[i] = (temp - of'(T[i])); temp = res - out
			dot(deltaBiases[i], false, nn->neurons[i - 1], true, nn->hfunc.f, NULL, NULL, NULL, &deltaWeights[i]); // dW[i] = <dB[i], f(transB(T[i-1]))>
		}
		else if (i == 0) {
			dot(nn->weights[i + 1], true, deltaBiases[i + 1], false, NULL, NULL, nn->neurons[i], nn->hfunc.df, &deltaBiases[i]); // dB[i] = <transA(W[i + 1]), dB[i + 1]> * f(T[i])
			dot(deltaBiases[i], false, inp, true, NULL, NULL, NULL, NULL, &deltaWeights[i]); // dW[i] = <dB[i], transB(inp)>
		}
		else {
			dot(nn->weights[i + 1], true, deltaBiases[i + 1], false, NULL, NULL, nn->neurons[i], nn->hfunc.df, &deltaBiases[i]); // dB[i] = <transA(W[i + 1]), dB[i + 1]> * f(T[i])
			dot(deltaBiases[i], false, nn->neurons[i - 1], true, nn->hfunc.f, NULL, NULL, NULL, &deltaWeights[i]); // dW[i] = <dB[i], f(transB(T[i-1]))>
		}
	}
	// calculating weights and biases
	for (int i = 0; i < nn->layers - 1; i++) {
		sub(nn->weights[i], deltaWeights[i], NULL, LearningRate, &temp); // W[i] - LearningRate * dW[i]
		copy_data(temp, &nn->weights[i]);
		if (nn->biases_on) {
			sub(nn->bias[i], deltaBiases[i], NULL, LearningRate, &temp); // B[i] - LearningRate * dB[i]
			copy_data(temp, &nn->bias[i]);
		}
	}
	free_matrix(&temp); // free temporary element
}
