#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
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

void check_result(double* bref, double* b, int size) {
	int i;
	bool equal = true;
	#pragma omp parallel private(i)
	{
		#pragma omp for
		for (i = 0; i < size * size; i++) {
			if (abs(bref[i] - b[i]) > 0.01 & equal == true)
			{
				printf("The first difference found is at the %ith iteration : \n", i + 1);
				printf("b_%i = %5.5f ", i, b[i]);
				printf("bref_%i = %5.5f ", i, bref[i]);			
				equal = false;
			}
		}
	}
	if (equal == true) printf("Result is ok!\n");
	else printf("Result is wrong!\n");
}

void matrixProduct(double* leftMatrix, double* rightMatrix,double* XMatrix, int size)
{
	int i,j,k;
	#pragma omp parallel private(i)
	{
		#pragma omp for
		for (i = 0; i < size * size; ++i)
			XMatrix[i] = 0.0;
	}
	
	#pragma omp parallel private(i,j,k)
	{
		#pragma omp for
		for (i = 0; i < size; ++i) 
			for (j = 0; j < size; ++j) 
				for (k = 0; k < size; ++k)
					XMatrix[i * size + j] += leftMatrix[i * size + k] * rightMatrix[k * size + j];
	}
}

void QRdecomposition(double* matrix, double* QMatrix, double* RMatrix, int size)
{
	int i,j,k;
	for (i = 0; i < size * size; ++i)
	{
		RMatrix[i] = 0.0;
		QMatrix[i] = 0.0;
	}
	for (i = 0; i < size; ++i)
	{
		double RMatrix_i_j = 0.0;

		for (j = 0; j < size; ++j)
			RMatrix_i_j += matrix[j * size + i] * matrix[j * size + i];

		RMatrix[i * (size + 1)] = sqrt(RMatrix_i_j);

		for (j = 0; j < size; ++j)
			QMatrix[j * size + i] = matrix[j * size + i] / RMatrix[i * (size + 1)];


		for (j = i + 1; j < size; ++j)
		{
			RMatrix_i_j = 0;

			for (k = 0; k < size; ++k)
				RMatrix_i_j += matrix[k * size + j] * QMatrix[k * size + i];

			RMatrix[i * size + j] = RMatrix_i_j;

			for (k = 0; k < size; ++k)
				matrix[k * size + j] -= RMatrix[i * size + j] * QMatrix[k * size + i];
		}
	}
}

//Function that solve the equation QRX=B where X is the unknown
void solveQRXeqB(double* QMatrix, double* RMatrix, double* XMatrix, double* BMatrix, int size)
{
	int i,j;
	double* transposedQMatrix = (double*)malloc(sizeof(double) * size * size);
	double* transposedQBMatrix = (double*)malloc(sizeof(double) * size * size);
	#pragma omp parallel private(i,j)
	{
		#pragma omp for
		for (i = 0; i < size * size; i++)
			XMatrix[i] = 0.0;

		
		#pragma omp for
		for (i = 0; i < size; i++) for (j = 0; j < size; j++)
				transposedQMatrix[i * size + j] = QMatrix[j * size + i];
	}	
	matrixProduct(transposedQMatrix, BMatrix, transposedQBMatrix, size);

	for (i = 0; i < size; ++i) {
		for (j = 0; j < size; ++j) {
			
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
	int size = 200;

	omp_set_num_threads(4);

	double* A = (double*)malloc(sizeof(double) * size * size);
	double* B = (double*)malloc(sizeof(double) * size * size);
	double* Bref = (double*)malloc(sizeof(double) * size * size);
	double* Q = (double*)malloc(sizeof(double) * size * size);
	double* R = (double*)malloc(sizeof(double) * size * size);
	double* QR = (double*)malloc(sizeof(double) * size * size);
	double* X = (double*)malloc(sizeof(double) * size * size);

	generate_matrix(A,size);
	generate_matrix(B,size);
	generate_matrix(Bref,size);

	int n = size, nrhs = size, lda = size, ldb = size, info;

	QRdecomposition(A, Q, R, size);

	solveQRXeqB(Q, R, X, B, size);
	
	matrixProduct(Q, R, QR, size);	

	matrixProduct(QR, X, B, size);
	
	check_result(Bref, B, size);

	free(Q);
	free(R);
	free(QR);
	free(X);
	free(A);
	free(Bref);
	free(B);
}
