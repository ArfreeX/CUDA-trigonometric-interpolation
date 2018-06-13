#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>


#define BLOCK_SIZE 8 // needs to be checked for proper values

__global__ void computeA(double* arrayA, double* arrayOfPoints, double* argumentsArray, int degree, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < degree + 1)
	{
		double sum = 0;

		for (int i = 0; i < size; i++)	// petla do zrownoleglenia
		{
			sum += (arrayOfPoints[size + i] * cos(x * argumentsArray[i]));
		}

		arrayA[x] = 2.0 / size * sum;
	}
}


__global__ void computeB(double* arrayB, double* arrayOfPoints, double* argumentsArray, int degree, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < degree + 1)
	{
		double sum = 0;

		for (int i = 0; i < size; i++)	// petla do zrownoleglenia
		{
			sum += (arrayOfPoints[size + i] * sin(x * argumentsArray[i]));
		}

		arrayB[x] = 2.0 / size * sum;
	}
}

void trigInterpolation(double *arrayOfPoints, int size)
{
	const long double PI = std::acos(-1.L);
	int degree = size / 2;
	
	double *arrayA, *arrayB, *argumentsArray;
	double *d_arrayOfPoints, *d_arrayA, *d_arrayB, *d_argumentsArray;

	arrayA = new double[degree+1];
	arrayB = new double[degree+1];

	argumentsArray = new double[size];

	


	int size_bytes = size * 2 * sizeof(double);		// number of bytes allocated on device mem
	int size_bytes_degree = degree + 1 * sizeof(double);
	int size_bytes_args = size * sizeof(double);
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 numBlocks((2 * size + BLOCK_SIZE - 1) / BLOCK_SIZE);


	// Cuda allocation
	auto err = cudaMalloc(&d_arrayOfPoints, size_bytes);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMalloc(&d_arrayA, size_bytes_degree);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMalloc(&d_arrayB, size_bytes_degree);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMalloc(&d_argumentsArray, size_bytes/2);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }




	// Arguments * 2Pi/size && filling values
	for (int i = 0; i < size; ++i)	
	{
		argumentsArray[i] = ((2*PI)/(double)size) * arrayOfPoints[i];
	}




	// Cuda data copy
	err = cudaMemcpy(d_arrayOfPoints, arrayOfPoints, size_bytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMemcpy(d_argumentsArray, argumentsArray, size_bytes_args, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMemcpy(d_arrayA, arrayA, size_bytes_degree, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	err = cudaMemcpy(d_arrayB, arrayB, size_bytes_degree, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	// Main algorithm code



	//============== CUDA ==============

	// Parameters A
	computeA << < numBlocks, threadsPerBlock >> > (d_arrayA, d_arrayOfPoints, d_argumentsArray, degree, size);

	err = cudaMemcpy(arrayA, d_arrayA, size_bytes_degree, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }



	// Parameters B	
	computeB << < numBlocks, threadsPerBlock >> > (d_arrayB, d_arrayOfPoints, d_argumentsArray, degree, size);

	err = cudaMemcpy(arrayB, d_arrayB, size_bytes_degree, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }


	//==================================
	// Results check

	std::cout << "\n\n";
	for (int i = 0; i < 3; i++)
		std::cout << arrayA[i] << "\n";
	for (int i = 0; i < 3; i++)
		std::cout << arrayB[i] << "\n";
	//==================================


	// wypisywanie algorytmu
	std::cout << "\nUzyskana funkcja interpolujaca \n\nG(x) = ";
	std::cout << arrayA[0] / 2.0;
	
	for (int i = 1; i < degree; i++)
	{
		if (arrayA[i] != 0)
		{
			std::cout << " + " << arrayA[i] << "*" << "cos( ";
			if (i == 1)	
				std::cout << "x )";
			else
				std::cout << i << "x )";
		}

		if (arrayB[i] != 0)
		{
			std::cout << " + " << arrayB[i] << "*" << "sin( ";
			if (i == 1)	
				std::cout << "x )";
			else
				std::cout << i << "x )";
		}
	}

	if (size % 2)
		std::cout << " + " << arrayA[degree]<< "*cos( " << degree << "x ) + " << arrayB[degree] << "sin( " << degree << "x )\n\n";
	else
		std::cout << " + " << arrayA[degree] / 2.0 << "*cos( " << degree << "x )\n\n";
	
	
	delete[] arrayA;
	delete[] arrayB;
	delete[] argumentsArray;
	cudaFree(d_argumentsArray);
	cudaFree(d_arrayA);
	cudaFree(d_arrayB);
	cudaFree(d_arrayOfPoints);
}

void showMatrix(double *array, int size)
{
	if (size > 0)
	{
		std::cout << "\n[  x ]: ";
		for (int i = 0; i < size; i++)
			std::cout << array[i] << "\t";

		std::cout << "\n[f(x)]: ";
		for (int i = size; i < size * 2; i++)
			std::cout << array[i] << "\t";
	}
}

double* readData(double* array, int &size)
{
	CFileStream file;
	std::cout << "Insert file path\n";
	file.openFile();
	array = file.readData(array, size);
	return array;
}

int main()
{
	double * arrayOfPoints = nullptr;
	int size = 0;
	arrayOfPoints = readData(arrayOfPoints, size);

	showMatrix(arrayOfPoints, size);

	trigInterpolation(arrayOfPoints, size);
	
	system("PAUSE");
	delete[] arrayOfPoints;
	return 0;
}