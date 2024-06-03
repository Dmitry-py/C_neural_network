#define _CRT_SECURE_NO_WARNINGS
#include "my_matrix.h"
#include "my_math.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void softmax(Matrix* m, Matrix** ans) {
	if (!(*ans)) { // if ans pointer is NULL that create new matrix object
		(*ans) = create_matrix(m->m, m->n, 0.0, NULL, false, 0.0, 0.0);
	}
	else if ((*ans)->m != m->m || (*ans)->n != m->n) { // update size of ans matrix and clear array
		(*ans)->m = m->m;
		(*ans)->n = m->n;
		free((*ans)->arr);
		(*ans)->arr = init_doubles((*ans)->m * (*ans)->n, true);
	}
	double sum = 0.0; // variable for sum of exponents of m elements
	for (int i = 0; i < m->m * m->n; i++) {
		sum += exp(m->arr[i]); // calculate the sum of exponents
	}
	for (int i = 0; i < m->m * m->n; i++) {
		(*ans)->arr[i] = exp(m->arr[i]) / sum; // calculate the softmax for every element in matrix m
	}
}

// memory allocating for doubles
double* init_doubles(int size, bool zeros) {
	double* res = (double*)malloc(size * sizeof(double));
	if (zeros) {
		for (int i = 0; i < size; i++) {
			res[i] = 0.0;
		}
	}
	return res;
}

// memory allocating for matrixes
Matrix** init_matrixes(int size, bool zeros) {
	Matrix** res = (Matrix**)malloc(size * sizeof(Matrix*));
	if (zeros) {
		for (int i = 0; i < size; i++) {
			res[i] = NULL;
		}
	}
	return res;
}

// print matrix
void printMatrix(Matrix* m, bool trans) {
	for (int i = 0; i < (trans ? m->n : m->m); i++) {
		for (int j = 0; j < (trans ? m->m : m->n); j++) {
			printf("%0.20g ", m->arr[m->n * (trans ? j : i) + (trans ? i : j)]);
		}
		printf("\n");
	}
}

// create matrix with m rows and n columns
// if with_randoms is true that initialize matrix with random values from [a, b]
// if data isn't NULL that copy data from "data"
// otherwise set each element with "val"
Matrix* create_matrix(int m, int n, double val, double* data, bool with_randoms, double a, double b) {
	Matrix* new = (Matrix*)malloc(sizeof(Matrix));
	new->arr = init_doubles(m * n, false); // allocate memory for data
	for (int i = 0; i < n * m; i++) {
		if (with_randoms) {
			new->arr[i] = get_rand_range(a, b); // getting random value from [a, b]
		}
		else if (data == NULL) {
			new->arr[i] = val;
		}
		else {
			new->arr[i] = data[i];
		}
	}
	new->m = m;
	new->n = n;
	return new;
}

// the function to use "func" on elems in the matrix m
void do_func_on_matrix(Matrix* m, my_func* func, Matrix** ans) {
	if (!(*ans)) { // if ans pointer is NULL that create new matrix object
		(*ans) = create_matrix(m->m, m->n, 0.0, NULL, false, 0.0, 0.0);
	}
	else if ((*ans)->m != m->m || (*ans)->n != m->n) { // update size of ans matrix and clear array
		(*ans)->m = m->m;
		(*ans)->n = m->n;
		free((*ans)->arr);
		(*ans)->arr = init_doubles((*ans)->m * (*ans)->n, true);
	}
	for (int i = 0; i < m->m * m->n; i++) {
		if (func) { // if "func" isn't NULL than doing function on matrix elem
			(*ans)->arr[i] = func(m->arr[i]);
		}
		else { // otherwise copying data
			(*ans)->arr[i] = m->arr[i];
		}
	}
}

// accuracy calculating as root mean square
double accuracy(Matrix* out, Matrix* m) {
	double res = 0.0;
	for (int i = 0; i < m->m; i++) {
		res += pow(out->arr[i] - m->arr[i], 2.0);
	}
	res /= (double)m->m;
	return res;
}

// the function to matrix multiplication
// the flag transA to transpose a (similar for transB)
// func_b contains function to activate b (similar for func_teta)
// complete formula for the function ans = (<T(a), T(func_b(b))> + alpha) * func_teta(teta)
void dot(Matrix* a, bool transA, Matrix* b, bool transB, my_func* func_b, Matrix* alpha, Matrix* teta, my_func* func_teta, Matrix** ans) {
	// similar "ans" initialization
	if (!(*ans)) {
		(*ans) = create_matrix((transA ? a->n : a->m), (transB ? b->m : b->n), 0.0, NULL, false, 0.0, 0.0);
	}
	else if ((*ans)->m != (transA ? a->n : a->m) || (*ans)->n != (transB ? b->m : b->n)) {
		(*ans)->m = (transA ? a->n : a->m);
		(*ans)->n = (transB ? b->m : b->n);
		free((*ans)->arr);
		(*ans)->arr = init_doubles((*ans)->m * (*ans)->n, true);
	}
	for (int i = 0; i < (transA ? a->n : a->m); i++) {
		for (int j = 0; j < (transB ? b->m : b->n); j++) {
			(*ans)->arr[(*ans)->n * i + j] = 0.0;
			for (int k = 0; k < (transA ? a->m : a->n); k++) {
				if (func_b) { // if func_b isn't NULL then perform the function on b element
					(*ans)->arr[(*ans)->n * i + j] += a->arr[a->n * (transA ? k : i) + (transA ? i : k)] * func_b(b->arr[b->n * (transB ? j : k) + (transB ? k : j)]);
				}
				else {
					(*ans)->arr[(*ans)->n * i + j] += a->arr[a->n * (transA ? k : i) + (transA ? i : k)] * b->arr[b->n * (transB ? j : k) + (transB ? k : j)];
				}
			}
			if (alpha) { // if alpha isn't NULL then adding an element from alpha
				(*ans)->arr[(*ans)->n * i + j] += alpha->arr[(*ans)->n * i + j];
			}
			if (teta) { // if teta isn't NULL then multiply an element from teta
				if (func_teta) { // if func_teta isn't NULL then perform the function on teta element
					(*ans)->arr[(*ans)->n * i + j] *= func_teta(teta->arr[(*ans)->n * i + j]);
				}
				else {
					(*ans)->arr[(*ans)->n * i + j] *= teta->arr[(*ans)->n * i + j];
				}
			}
		}
	}
}

// the function to element-wise matrix subtraction
void sub(Matrix* a, Matrix* b, my_func* func_b, double alpha, Matrix** ans) {
	// similar "ans" initialization
	if (!(*ans)) {
		(*ans) = create_matrix(a->m, a->n, 0.0, NULL, false, 0.0, 0.0);
	}
	else if (((*ans)->m != a->m) || ((*ans)->n != a->n)) {
		free_matrix(ans);
		(*ans) = create_matrix(a->m, a->n, 0.0, NULL, false, 0.0, 0.0);
	}
	for (int i = 0; i < a->m * a->n; i++) {
		double temp_b = b->arr[i];
		if (func_b) { // if func_b isn't NULL then perform the function on b element
			temp_b = func_b(temp_b);
		}
		(*ans)->arr[i] = a->arr[i] - temp_b * alpha;
	}
}

// the function to element-wise matrix multiplication
void mul(Matrix* a, Matrix* b, my_func* func_b, double alpha, Matrix** ans) {
	// similar "ans" initialization
	if (!(*ans)) {
		(*ans) = create_matrix(a->m, a->n, 0.0, NULL, false, 0.0, 0.0);
	}
	else if ((*ans)->m != a->m || (*ans)->n != a->n) {
		(*ans)->m = a->m;
		(*ans)->n = a->n;
		free((*ans)->arr);
		(*ans)->arr = init_doubles(a->m * a->n, true);
	}
	for (int i = 0; i < a->m * a->n; i++) {
		double temp_b = b->arr[i];
		if (func_b) { // if func_b isn't NULL then perform the function on b element
			temp_b = func_b(temp_b);
		}
		(*ans)->arr[i] = a->arr[i] * temp_b * alpha;
	}
}

// the function to copy elements from a to b
void copy_data(Matrix* a, Matrix** b) {
	// similar "b" initialization
	if (!(*b)) {
		(*b) = create_matrix(a->m, a->n, 0.0, NULL, false, 0.0, 0.0);
	}
	else if ((*b)->m != a->m || (*b)->n != a->n) {
		(*b)->m = a->m;
		(*b)->n = a->n;
		free((*b)->arr);
		(*b)->arr = init_doubles(a->m * a->n, true);
	}
	for (int i = 0; i < a->m * a->n; i++) {
		(*b)->arr[i] = a->arr[i];
	}
}

void free_matrix(Matrix** m) { // clear matrix m
	if (!m) return;
	if (!(*m)) return;
	free((*m)->arr);
	free((*m));
	*m = NULL;
}
