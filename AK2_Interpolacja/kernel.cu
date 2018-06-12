#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


void trigInterpolation(double *arrayOfPoints, int size)
{
	const double PI = 3.14159265358979323846264338327950288419;

	int degree = size / 2;
	
	double *arrayA, *arrayB, *argumentsArray;

	arrayA = new double[degree+1];
	arrayB = new double[degree+1];
	argumentsArray = new double[size];
	
	for (int i = 0; i < size; ++i)	// nowa tablica, gdzie mnozenie argumentow przez 2pi/size
	{
		argumentsArray[i] = ((2*PI)/(double)size) * arrayOfPoints[i];
	}


	// Main algorithm code
	for (int i = 0; i < degree+1; i++)
	{
		double sum = 0;		

		for (int j = 0; j < size; j++)	// petla do zrownoleglenia
		{
			sum += ( arrayOfPoints[size + j] * cos( (double)i * argumentsArray[ j ] ) );
		}

		arrayA[i] = 2.0 / size * sum;
	}

	for (int i = 0; i < degree + 1; i++)
	{
		double sum = 0;

		for (int j = 0; j < size; j++) // petla do zrownoleglenia
		{
			sum += (arrayOfPoints[size + j] * sin((double)i * argumentsArray[ j ] ) );
		}

		arrayB[i] = 2.0 / size * sum;
	}

	//==================================
	// Test poprawnosci wynikow
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
			if (i == 1)		// XD
				std::cout << "x )";
			else
				std::cout << i << "x )";
		}

		if (arrayB[i] != 0)
		{
			std::cout << " + " << arrayB[i] << "*" << "sin( ";
			if (i == 1)		// XD * XD
				std::cout << "x )";
			else
				std::cout << i << "x )";
		}
	}

	std::cout << " + " << arrayA[degree] / 2.0 << "*cos( " << degree << "x )\n\n";
	
	
	delete[] arrayA;
	delete[] arrayB;
	delete[] argumentsArray;
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



