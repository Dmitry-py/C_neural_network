#pragma once
#include "my_math.h"
#include <stdbool.h>

// the structure to contain double matrix with m rows and n columns
typedef struct Matrix_s {
	double* arr;
	int m, n;
} Matrix;

// softmax function
void softmax(Matrix* m, Matrix** ans);

// functions for allocating memory for double and matrix objects
// size - count of elements, zeros - flag to set zero elements in the array
double* init_doubles(int size, bool zeros);
Matrix** init_matrixes(int size, bool zeros);

// the function to print matrix m, trans - flag to transpose output matrix
void printMatrix(Matrix* m, bool trans);
// the function to initialize matrix with m rows and n columns
Matrix* create_matrix(int m, int n, double val, double* data, bool with_randoms, double a, double b);
// the function to use "func" on elems in the matrix m
void do_func_on_matrix(Matrix* m, my_func* func, Matrix** ans);
// the function to calculating accuracy
double accuracy(Matrix* out, Matrix* m);
// the function to matrix multiplication
// more description in the *.c file
void dot(Matrix* a, bool transA, Matrix* b, bool transB, my_func* func_b, Matrix* alpha, Matrix* teta, my_func* func_teta, Matrix** ans);
// the function to element-wise matrix subtraction
void sub(Matrix* a, Matrix* b, my_func* func_b, double alpha, Matrix** ans);
// the function to element-wise matrix multiplication
void mul(Matrix* a, Matrix* b, my_func* func_b, double alpha, Matrix** ans);
// the function to copy elements from a to b
void copy_data(Matrix* a, Matrix** b);
// the function to free matrix m
void free_matrix(Matrix** m);
