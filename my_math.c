#include "my_math.h"
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

bool flag = true;

double get_rand() {
	if (flag) {
		srand(time(NULL)); // set seed by time for rand function
		flag = false;
	}
	return (double)rand() / (double)RAND_MAX; // reduce the random number to a value from 0 to 1
}

double get_rand_range(double a, double b) {
	if (flag) {
		srand(time(NULL)); // set seed by time for rand function
		flag = false;
	}
	return get_rand() * (b - a) + a;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
}

double ELU(double x) {
	return (x > 0.0 ? x : (exp(x) - 1));
}

double dELU(double x) {
	return (x > 0.0 ? 1.0 : exp(x));
}

double SiLU(double x) {
	return x * sigmoid(x);
}

double dSiLU(double x) {
	return dsigmoid(x) * (exp(x) + x + 1.0);
}

double dtanh(double x) {
	return 1.0 - pow(tanh(x), 2.0);
}

double ReLU(double x) {
	return (x >= 0.0) ? x : 0.0;
}

double dReLU(double x) {
	return (x >= 0.0) ? 1.0 : 0.0;
}

double linear(double x) {
	return x;
}

double dlinear(double x) {
	return 1.0;
}

 // returns a linear function by default
function_set get_func(int key) {
	function_set res;
	res.type = key;
	switch (key) {
	case 0:
		res.f = sigmoid;
		res.df = dsigmoid;
		break;
	case 1:
		res.f = ELU;
		res.df = dELU;
		break;
	case 2:
		res.f = SiLU;
		res.df = dSiLU;
		break;
	case 3:
		res.f = tanh;
		res.df = dtanh;
		break;
	case 4:
		res.f = ReLU;
		res.df = dReLU;
		break;
	default:
		res.df = dlinear;
		res.f = linear;
		res.type = -1;
		break;
	}
	return res;
}
