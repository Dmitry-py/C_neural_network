#pragma once

// my type for activation func
typedef double my_func(double);

// structurte that contains pointers of function and derivative of a function
// and num of activation function
typedef struct function_set_s {
	int type;
	my_func* f;
	my_func* df;
} function_set;

// the function to getting a random number in the range from 0 to 1
double get_rand();
// the function to getting a random number in the range from a to b
double get_rand_range(double a, double b);

// activation functions
double sigmoid(double x);
double dsigmoid(double x);
double ELU(double x);
double dELU(double x);
double SiLU(double x);
double dSiLU(double x);
// tanh from math.h
double dtanh(double x);
double ReLU(double x);
double dReLU(double x);
double linear(double x);
double dlinear(double x);

// the function that returns a function_set by key
function_set get_func(int key);
