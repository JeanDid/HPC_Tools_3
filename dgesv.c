#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
# include <omp.h>
//#include "mkl_lapacke.h"

void generate_matrix(double* matrix, int size)
{
	int i;
	srand(1);

	for (i = 0; i < size * size; i++)
	{
		matrix[i] = rand() % 100;
	}
}

void print_matrix(const char* name, double* matrix, int size)
{
	int i, j;
	printf("\n\nmatrix %s : \n \n", name);

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			printf("%f ", matrix[i * size + j]);
		}
		printf("\n");
	}
}

int check_result(double* bref, double* b, int size) {
	int i;
	for (i = 0; i < size * size; i++) {
		if (abs(bref[i] - b[i]) > 0.01)
		{
			printf("The first difference found is at the %ith iteration : \n", i + 1);
			printf("b_%i = %5.30f ", i, b[i]);
			printf("bref_%i = %5.30f ", i, bref[i]);
			return 0;
		}
	}
	return 1;
}

void matrixProduct(double* leftMatrix, double* rightMatrix,double* XMatrix, int size)
{
	for (int i = 0; i < size * size; ++i) 
		XMatrix[i] = 0.0;

	for (int i = 0; i < size; ++i) 
		for (int j = 0; j < size; ++j) 
			for (int k = 0; k < size; ++k) 
		XMatrix[i * size + j] += leftMatrix[i * size + k] * rightMatrix[k * size + j];
}

void QRdecomposition(double* matrix, double* QMatrix, double* RMatrix, int size)
{
	for (int i = 0; i < size * size; ++i)
	{
		RMatrix[i] = 0.0;
		QMatrix[i] = 0.0;
	}

	for (int i = 0; i < size; ++i)
	{
		double RMatrix_i_j = 0.0;

		for (int j = 0; j < size; ++j)
			RMatrix_i_j += matrix[j * size + i] * matrix[j * size + i];

		RMatrix[i * (size + 1)] = sqrt(RMatrix_i_j);

		for (int j = 0; j < size; ++j)
			QMatrix[j * size + i] = matrix[j * size + i] / RMatrix[i * (size + 1)];

		for (int j = i + 1; j < size; ++j)
		{
			RMatrix_i_j = 0;

			for (int k = 0; k < size; ++k)
				RMatrix_i_j += matrix[k * size + j] * QMatrix[k * size + i];

			RMatrix[i * size + j] = RMatrix_i_j;

			for (int k = 0; k < size; ++k)
				matrix[k * size + j] -= RMatrix[i * size + j] * QMatrix[k * size + i];
		}
	}
}

void solveQRXeqB(double* QMatrix, double* RMatrix, double* XMatrix, double* BMatrix, int size)
{
	for (int i = 0; i < size * size; ++i) XMatrix[i] = 0.0;

	double* transposedQMatrix = (double*)malloc(sizeof(double) * size * size);
	
	for (int i = 0; i < size; ++i) for (int j = 0; j < size; ++j)
			transposedQMatrix[i * size + j] = QMatrix[j * size + i];
	
	double* transposedQBMatrix = (double*)malloc(sizeof(double) * size * size);

	matrixProduct(transposedQMatrix, BMatrix, transposedQBMatrix, size);

	printf("\n");

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			double sou = 0.0;

			for (int sub = size - i; sub < size; ++sub) {
				sou += RMatrix[(size - i - 1) * size + sub] * XMatrix[sub * size + (size - j - 1)];
			}
			XMatrix[(size - i - 1) * size + (size - j - 1)] = (transposedQBMatrix[(size - i - 1) * size + (size - j - 1)] - sou) / RMatrix[(size - i - 1) * (size + 1)];
		}
	}
	free(transposedQMatrix);
	free(transposedQBMatrix);
}

void main(int argc, char* argv[])
{

	//int size = atoi(argv[1]);
	int size = 500;

	
	double* a = (double*)malloc(sizeof(double) * size * size);
	double* b = (double*)malloc(sizeof(double) * size * size);
	double* bref = (double*)malloc(sizeof(double) * size * size);
	double* QR = (double*)malloc(sizeof(double) * size * size);

	generate_matrix(a,size);
	generate_matrix(b,size);
	generate_matrix(bref,size);

	int n = size, nrhs = size, lda = size, ldb = size, info;


	double* Q = (double*)malloc(sizeof(double) * size * size);
	double* R = (double*)malloc(sizeof(double) * size * size);


	QRdecomposition(a, Q, R, size);



	double* X = (double*)malloc(sizeof(double) * size * size);

	solveQRXeqB(Q, R, X, b, size);
	
	matrixProduct(Q, R, QR, size);	

	matrixProduct(QR, X, b, size);

	free(Q);
	free(R);
	free(X);
	free(a);
	
	if (check_result(bref, b, size) == 1)
		printf("Result is ok!\n");
	else
		printf("Result is wrong!\n");

	free(bref);
	free(b);
	free(QR);
}
